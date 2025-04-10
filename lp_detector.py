import argparse
import os
import cv2
import re 

import numpy as np
from glob import glob
from tqdm import tqdm
from time import time
import torch
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append("./")
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
import warnings
warnings.filterwarnings("ignore")


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    font = ImageFont.load_default()

    # --- Draw predicted boxes ---
    for box, label in zip(boxes, labels):
        # Convert from [cx, cy, w, h] (0-1) to [x0, y0, x1, y1] in image size
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = box.int().tolist()

        # Random color for each prediction
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        label_text = str(label)
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), label_text, font)
        else:
            w, h = draw.textsize(label_text, font)
            bbox = (x0, y0, x0 + w, y0 + h)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), label_text, fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def filter_large_bboxes_with_phrases(boxes, phrases, max_area=0.3):

    areas = boxes[:, 2] * boxes[:, 3]
    mask = areas < max_area
    filtered_boxes = boxes[mask]
    filtered_phrases = [p for i, p in enumerate(phrases) if mask[i]]

    return filtered_boxes, filtered_phrases

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


def crop_license_plate(image, bbox, ocr_model=None):
    """
    image: numpy array (H, W, 3)
    bbox: torch.Tensor of shape (4,) in [cx, cy, w, h] normalized format
    ocr_model: Optional OCR model. If None, will use Tesseract.
    """
    H, W, _ = image.shape    
    cx, cy, bw, bh = bbox[0].tolist()

    # Convert to pixel values
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)

    # Ensure coordinates are within bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    cropped = image[y1:y2, x1:x2]

    # OCR part
    if ocr_model:
        text = ocr_model(cropped)  # Placeholder for custom model inference
    else:
        import pytesseract
        text = pytesseract.image_to_string(cropped, config='--psm 7')  # PSM 7: treat image as a single text line

    return cropped, text.strip()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, default=None, help="path to image file")
    parser.add_argument("--image_dir", "-id", default=None, type=str, help="path to image dir")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.3, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_paths = [args.image_path]
    image_dir = args.image_dir
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    output_dir = f"{args.output_dir}_threshold_{box_threshold}"
    
    token_spans = args.token_spans
    im_files = [".png", ".jpg", ".jpeg", ".bmp"]
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # if image_dir: image_paths = glob(f"{image_dir}/*/*{[im_file for im_file in im_files]}")    
    if image_dir: image_paths = glob(f"{image_dir}/*.png")    
    start = time()
    for idx, image_path in tqdm(enumerate(image_paths), desc = "Detecting objects..."):       
        print(f"image_path -> {image_path}")
        cv_image = cv2.imread(image_path) 
        # if not os.path.basename(image_path) in ["000000ad_72704_197632.jpg", "000000ad_102400_191488.jpg", "000009ac_98304_197632.jpg", "00000a1d_19456_236544.jpg"]: continue
        # if not os.path.basename(image_path) in ["00000a1d_19456_236544.jpg"]: continue
        if idx == 20: break
        fname = os.path.splitext(os.path.basename(image_path))[0]        
        # load image        
        image_pil, image = load_image(image_path)        
        # image_pil.save(fp = f"{output_dir}/{fname}_original.jpg")
        # Image.fromarray((image*255).permute(1,2,0).numpy().astype(np.uint8)).save(fp = f"{output_dir}/{fname}_tfs.jpg")        
        
        # set the text_threshold to None if token_spans is set.
        if token_spans is not None:
            text_threshold = None
            print("Using token_spans. Set the text_threshold to None.")

        # run model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
        )        
        
        cropped_lp, plate_text = crop_license_plate(cv_image, boxes_filt)

        print("License Plate Text:", plate_text)

        cleaned_text = re.sub(r'[^A-Za-z0-9]', '', plate_text)
        print(f"cleaned_text -> {cleaned_text}")

        # # visualize pred
        # size = image_pil.size
        # pred_dict = {
        #     "boxes": boxes_filt,
        #     "size": [size[1], size[0]],  # H,W
        #     "labels": pred_phrases,
        # }
        # # import ipdb; ipdb.set_trace()
        # image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        # image_with_box.save(os.path.join(output_dir, f"{fname}_DINO_detection.jpg"))
    print(f"Inference is done in {(time() - start)} secs!")