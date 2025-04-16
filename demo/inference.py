import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from torchvision.ops import nms
import sys
sys.path.append("./")
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T

class GroundingDINOHandler:
    def __init__(self, model_config_path, model_checkpoint_path, device):
        self.device = device
        self.model = self.load_model(model_config_path, model_checkpoint_path)
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_config_path, model_checkpoint_path):
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model.to(self.device)

    def load_image(self, image_path):
        image_pil = Image.open(image_path).convert("RGB")
        image, _ = self.transform(image_pil, None)
        return image_pil, image

    def plot_boxes_to_image(self, image_pil, tgt, lbl_path):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        font = ImageFont.load_default()

        for box, label in zip(boxes, labels):
            box = box.to("cpu")
            box = box * torch.Tensor([W, H, W, H])            
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.int().tolist()
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            label_text = str(label)
            bbox = draw.textbbox((x0, y0), label_text, font) if hasattr(draw, "textbbox") else (x0, y0, x0 + 50, y0 + 10)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), label_text, fill="white", font=font)
            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cx, cy, w, h = map(float, parts[-4:])
                class_name = " ".join(parts[:-4])
                

                if class_name in ["normal", "abnormal cell"]: continue                
                
                x0 = int((cx - w / 2) * W)
                y0 = int((cy - h / 2) * H)
                x1 = int((cx + w / 2) * W)
                y1 = int((cy + h / 2) * H)
                draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
                text = f"GT: {class_name}"
                bbox = draw.textbbox((x0, y0), text, font) if hasattr(draw, "textbbox") else (x0, y0, x0 + 50, y0 + 10)
                draw.rectangle(bbox, fill='red')
                draw.text((x0, y0), text, fill="white", font=font)

        return image_pil, mask

    def get_grounding_output(self, image, caption, box_threshold, text_threshold=None, with_logits=True, token_spans=None):
        assert text_threshold is not None or token_spans is not None
        caption = caption.lower().strip()
        caption = caption if caption.endswith(".") else caption + "."
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        if token_spans is None:
            filt_mask = logits.max(dim=1)[0] > box_threshold
            logits_filt = logits[filt_mask]
            boxes_filt = boxes[filt_mask]
            tokenized = self.model.tokenizer(caption)
            pred_phrases = [get_phrases_from_posmap(logit > text_threshold, tokenized, self.model.tokenizer) +
                            (f"({str(logit.max().item())[:4]})" if with_logits else "")
                            for logit in logits_filt]
        else:
            positive_maps = create_positive_map_from_span(
                self.model.tokenizer(caption),
                token_span=token_spans
            ).to(image.device)
            logits_for_phrases = positive_maps @ logits.T
            all_boxes, all_phrases = [], []
            for token_span, logit_phr in zip(token_spans, logits_for_phrases):
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                filt_mask = logit_phr > box_threshold
                all_boxes.append(boxes[filt_mask])
                if with_logits:
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr[filt_mask]])
                else:
                    all_phrases.extend([phrase] * filt_mask.sum().item())
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases

        return boxes_filt, pred_phrases

    @staticmethod
    def xywh_to_xyxy(boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    @staticmethod
    def extract_confidences(pred_phrases):
        return torch.tensor([float(p.split('(')[-1].rstrip(')')) if '(' in p else 1.0 for p in pred_phrases], device=device)

    def nms_by_confidence(self, boxes, phrases, iou_threshold=0.2, max_box_size=0.8):
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        size_mask = (widths <= max_box_size) & (heights <= max_box_size)
        boxes = boxes[size_mask]
        phrases = [phrases[i] for i in torch.where(size_mask)[0]]

        if len(boxes) == 0: return boxes, []

        confidences = self.extract_confidences(phrases)
        boxes_xyxy = self.xywh_to_xyxy(boxes)        
        keep = nms(boxes_xyxy, confidences, iou_threshold)
        return boxes[keep], [phrases[i] for i in keep]

    @staticmethod
    def filter_large_bboxes_with_phrases(boxes, phrases, max_area=0.3):
        areas = boxes[:, 2] * boxes[:, 3]
        mask = areas < max_area
        filtered_boxes = boxes[mask]
        filtered_phrases = [p for i, p in enumerate(phrases) if mask[i]]
        return filtered_boxes, filtered_phrases

    @staticmethod
    def compute_iou(box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0
    
import os
import torch
import tifffile as tiff
import timm
import pickle
import time
import numpy as np
import argparse
from torchvision import transforms as TS
from PIL import Image
from tqdm import tqdm
from roi_lbc import RoiLBC

class Inference:
    def __init__(self, results_output_dir, device, save_model_path, save_data_path, im_size,
                 slide_type, data_name, run_name, project_type):        
        
        self.slide_type = slide_type
        self.im_size    = im_size
        save_name = "beki" if "high" in save_model_path else "baseline"
        self.results_output_dir = f"{results_output_dir}/{data_name}_{save_name}"        
        self.device = device        
        self.model_names = ["resnet50", "rexnet_150", "resnext50_32x4d"]
        self.ckpt_paths = [ f"{save_model_path}/{data_name}_{run_name}_{model_name}_{project_type}_best_model.pth" for model_name in self.model_names ]

        with open(f"{save_data_path}/{data_name}_{run_name}_{project_type}_{self.im_size}_cls_names.pkl", "rb") as fp: self.classes = pickle.load(fp)
        print(self.classes)

    def get_tfs(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        return TS.Compose([TS.Resize((self.im_size, self.im_size)), TS.ToTensor(), TS.Normalize(mean=mean, std=std)])

    def load_models(self):
        self.models_di = {}
        for i, (model_name, ckpt_name) in enumerate(zip(self.model_names, self.ckpt_paths)):
            model = timm.create_model(model_name=model_name, num_classes=len(self.classes))
            if "beki" in self.results_output_dir: model.load_state_dict(torch.load(ckpt_name, weights_only = False)["model_state_dict"])
            else: model.load_state_dict(torch.load(ckpt_name, weights_only = False))
            model.eval()
            self.models_di[f"model_{i}"] = model.to(self.device)

    def get_roi(self):
        
        tile, size_0, size_8 = self.read_slide()
        roi = RoiLBC()
        bbox = roi.main_method(tile)
        print(f"Bounding Box: {bbox}")
        
        original_width = size_0[0]
        original_height = size_0[1]
        resized_width = size_8[0]
        resized_height = size_8[1]
        x_ratio = original_width / resized_width
        y_ratio = original_height / resized_height
        
        bbox = [int(bbox[0] * x_ratio), int(bbox[1] * y_ratio), int((bbox[0]+bbox[2]) * x_ratio), int((bbox[1]+(bbox[3])) * y_ratio)]

        print(f"Original Bounding Box: {bbox}")

        return bbox   
    
    def get_im(self, crop): return self.get_tfs()(crop).unsqueeze(dim=0)

    def read_slide(self):
        
        # tiff path file
        with tiff.TiffFile(self.tiff_path) as slide:

            # slid 2 size
            size_0 = slide.pages[0].shape
            print(f"Original slide size: {size_0}")
            size_8 = slide.pages[8].shape
            print(f"8th layer slide size: {size_8}")            
        
            tile = slide.pages[8].asarray()
            tile = Image.fromarray(tile)
            tile = tile.convert("RGB")           

            original_image = tile.copy()         

        return original_image, size_0, size_8
    
    def bgr2rgb(self, image_array):
        image = Image.fromarray(image_array.astype(np.uint8))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def process_tiff(self, crop, fname, save_name):
        self.load_models()
        os.makedirs(self.results_output_dir, exist_ok=True)       
        
        save_folder = os.path.join(self.results_output_dir, fname)        
        abnormal_dir = f"{save_folder}/abnormal"
        normal_dir = f"{save_folder}/normal"         
        os.makedirs(abnormal_dir, exist_ok=True); os.makedirs(normal_dir, exist_ok=True)

        crop = self.bgr2rgb(crop)
        im = self.get_im(crop).to(self.device)
        predictions = []

        with torch.no_grad():
            for i in range(len(self.models_di)):
                pred = self.models_di[f"model_{i}"](im)
                predictions.append(pred)

        ensemble_pred = torch.argmax(torch.sum(torch.stack(predictions), dim=0), dim=1).item()
        
        class_name = list(self.classes.keys())[ensemble_pred]        
        save_file_name = os.path.join(abnormal_dir, save_name) if class_name == "abnormal" else os.path.join(normal_dir, save_name)        
        crop.save(save_file_name)            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("--results_output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_model_path", required=True)
    parser.add_argument("--save_data_path", required=True)
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--slide_type", required=True)
    parser.add_argument("--project_type", required=True)
    parser.add_argument("--im_size", type=int, required=True)    

    args = parser.parse_args()    
    device = args.device
    
    start_time = time.time()
    
    inf = Inference(        
        results_output_dir=args.results_output_dir,
        device=args.device,
        save_model_path=args.save_model_path,
        save_data_path=args.save_data_path,
        data_name=args.data_name,
        run_name=args.run_name,
        slide_type=args.slide_type,
        project_type=args.project_type,
        im_size=args.im_size

    )   

    tfs = inf.get_tfs()

    handler = GroundingDINOHandler("/home/bekhzod/Desktop/localization_models_performance/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../weights/groundingdino_swint_ogc.pth", device=device)

    image_dir   = "/mnt/data/cervical_screening/classification/hospital_dataset/classification/tile_classification/25_04_12_test_PAP/abnormal"
    lbls_dir    = "/mnt/data/cervical_screening/classification/hospital_dataset/25_04_02_1024"
    output_dir  = "../추론_결과_dellll"
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(f"{image_dir}/*.jpg")    
    for idx, image_path in enumerate(image_paths):
        if idx == 20: break
        fname = os.path.splitext(os.path.basename(image_path))[0].split("_")[0]
        dirname  = os.path.basename(image_path).split("_")[0]
        lbl_path = f"{lbls_dir}/{dirname}/{os.path.basename(image_path).replace('.jpg', '.txt')}"

        image_pil, image_tensor = handler.load_image(image_path)
        boxes, phrases = handler.get_grounding_output(image_tensor, caption = "cells", box_threshold=0.05, text_threshold=0.05)
        boxes, phrases = handler.nms_by_confidence(boxes = boxes, phrases = phrases, iou_threshold=0.2, max_box_size=0.8)

        size = image_pil.size
        # pred_dict = {
        #     "boxes": boxes,
        #     "size": [size[1], size[0]],  # H,W
        #     "labels": phrases,
        # }


        # # import ipdb; ipdb.set_trace()
        # image_with_box = handler.plot_boxes_to_image(image_pil, pred_dict, lbl_path=lbl_path)[0]
        # image_with_box.save(os.path.join(output_dir, f"{fname}.jpg")) 


        w, h = image_pil.size

        crops = []
        for i, (xc, yc, bw, bh) in tqdm(enumerate(boxes), desc=f"Processing crops of {fname}..."):
            # Convert from normalized center format to pixel corner format
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            # Ensure coordinates are within image boundaries
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, w)
            y2 = min(y2, h)

            # Crop and save
            crop = np.array(image_pil.crop((x1, y1, x2, y2)))
            # im_tensor = tfs(crop)
            # print(im_tensor.shape)
            inf.process_tiff(crop=crop, fname=fname, save_name = f"{fname}_crop_{i}.jpg")
            # crop.save(f"{output_dir}/{fname}_crop_{i}.jpg")
        
    print(f"\nInference completed in {(time.time() - start_time):.1f} seconds.")

    

        