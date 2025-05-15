import argparse
import os


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


def plot_boxes_to_image(image_pil, tgt, lbl_path):
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

    # --- Draw GT boxes from YOLO label file ---    
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # skip invalid lines

        # Last 4 are bbox values
        cx, cy, w, h = map(float, parts[-4:])
        class_name = " ".join(parts[:-4])
        if class_name in ["normal", "abnormal cell"]: continue

        x0 = int((cx - w / 2) * W)
        y0 = int((cy - h / 2) * H)
        x1 = int((cx + w / 2) * W)
        y1 = int((cy + h / 2) * H)

        draw.rectangle([x0, y0, x1, y1], outline='red', width=3)

        text = f"GT: {class_name}"
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), text, font)
        else:
            w_text, h_text = draw.textsize(text, font)
            bbox = (x0, y0, x0 + w_text, y0 + h_text)

        draw.rectangle(bbox, fill='red')
        draw.text((x0, y0), text, fill="white", font=font)
    # except Exception as e:
    #     print(f"Failed to load GT labels from {lbl_path}: {e}")

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


from torchvision.ops import nms

def parse_confidences(pred_phrases):
    """Extract confidence scores from strings like 'cells(0.67)'."""
    scores = []
    for phrase in pred_phrases:
        try:
            conf = float(phrase.split('(')[-1].replace(')', ''))
        except:
            conf = 1.0  # fallback
        scores.append(conf)
    return torch.tensor(scores)

# def apply_nms_and_filter_boxes(boxes_filt, pred_phrases, iou_thresh=0.7, max_box_size=0.8):
#     """
#     Apply NMS and filter out boxes that are too large.
#     """
#     # Remove large boxes BEFORE NMS
#     widths = boxes_filt[:, 2]
#     heights = boxes_filt[:, 3]
#     size_mask = (widths <= max_box_size) & (heights <= max_box_size)
    
#     boxes_filt = boxes_filt[size_mask]
#     pred_phrases = [pred_phrases[i] for i in torch.where(size_mask)[0]]

#     if len(boxes_filt) == 0:
#         return boxes_filt, []

#     # Convert cx, cy, w, h -> x1, y1, x2, y2
#     boxes = boxes_filt.clone()
#     boxes_xyxy = torch.zeros_like(boxes)
#     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
#     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
#     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
#     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

#     scores = parse_confidences(pred_phrases)

#     # Apply NMS
#     keep_indices = nms(boxes_xyxy, scores, iou_threshold=iou_thresh)

#     filtered_boxes = boxes_filt[keep_indices]
#     filtered_phrases = [pred_phrases[i] for i in keep_indices]

#     return filtered_boxes, filtered_phrases

def extract_confidences(pred_phrases):
    return torch.tensor([float(p.split('(')[-1].rstrip(')')) for p in pred_phrases])

def xywh_to_xyxy(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def nms_by_confidence(boxes, phrases, iou_threshold=0., max_box_size = 0.9):

    widths = boxes[:, 2]
    heights = boxes[:, 3]
    size_mask = (widths <= max_box_size) & (heights <= max_box_size)
    
    boxes = boxes[size_mask]
    phrases = [phrases[i] for i in torch.where(size_mask)[0]]

    if len(boxes) == 0:
        return boxes, []
    
    confidences = extract_confidences(phrases)
    boxes_xyxy = xywh_to_xyxy(boxes)
    indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)

    keep = []
    removed = set()

    for i in indices:
        if i in removed:
            continue
        keep.append(i)
        for j in indices:
            if j != i and j not in removed:
                iou = compute_iou(boxes_xyxy[i], boxes_xyxy[j])
                if iou > iou_threshold:
                    removed.add(j)

    filtered_boxes = boxes[keep]
    filtered_phrases = [phrases[i] for i in keep]
    return filtered_boxes, filtered_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, default=None, help="path to image file")
    # parser.add_argument("--image_dir", "-id", default=None, type=str, help="path to image dir")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.05, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.05, help="text threshold")
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
    lbls_dir = "/mnt/data/cervical_screening/classification/hospital_dataset/25_05_07_test"
    image_paths = glob(f"{lbls_dir}/*/abnormal/*.jpg")
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    output_dir = f"{args.output_dir}_threshold_{box_threshold}"
    
    token_spans = args.token_spans
    im_files = [".png", ".jpg", ".jpeg", ".bmp"]
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # lbls_dir = "/mnt/data/cervical_screening/classification/hospital_dataset/25_04_02_1024"
    
    
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

      
    # if image_dir: image_paths = glob(f"{image_dir}/*.jpg")    
    start = time()
    for idx, image_path in tqdm(enumerate(image_paths), desc = "Detecting objects..."):         
        start_time = time()       
        type = os.path.basename(image_path).split("_")[0] 
        dirname  = os.path.basename(image_path).split()[-1].split("_")[2]
        lbl_path = f"{lbls_dir}/{dirname}/{os.path.basename(image_path.split(f'{type}_{type}_')[-1]).replace('.jpg', '.txt')}"
        # lbl_path = image_path.replace(".jpg", ".txt")        

        # if not os.path.basename(image_path) in ["000000ad_72704_197632.jpg", "000000ad_102400_191488.jpg", "000009ac_98304_197632.jpg", "00000a1d_19456_236544.jpg"]: continue
        # if not os.path.basename(image_path) in ["00000a1d_19456_236544.jpg"]: continue
        # if idx == 205: break
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

        # print(f"before boxes_filt-> {boxes_filt}")
        # print(f"before pred_phrases -> {pred_phrases}")
        # boxes_filt, pred_phrases = filter_large_bboxes_with_phrases(boxes_filt, pred_phrases)        
        # boxes_filt, pred_phrases = apply_nms_and_filter_boxes(boxes_filt, pred_phrases, iou_thresh=0.5)
        boxes_filt, pred_phrases = nms_by_confidence(boxes_filt, pred_phrases, iou_threshold=0.05)
        print(f"Inference is done in -> {time() - start_time:.3f} secs")
        # print(f"after boxes_filt-> {boxes_filt}")
        # print(f"after pred_phrases -> {pred_phrases}")

        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        # import ipdb; ipdb.set_trace()
        image_with_box = plot_boxes_to_image(image_pil, pred_dict, lbl_path=lbl_path)[0]
        image_with_box.save(os.path.join(output_dir, f"{fname}_DINO_detection.jpg"))
    print(f"Inference is done in {(time() - start)} secs!")