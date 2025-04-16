import os
import random
import re
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import urllib.request

import cv2
import pytesseract
import torch
from glob import glob

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.slconfig import SLConfig

class GroundingDINOApp:
    def __init__(self, config_path, checkpoint_path, device, cpu_only=True):
        self.cpu_only = cpu_only        
        self.device   = device
        self.model = self.load_model(config_path, checkpoint_path)

    def load_model(self, config_path, checkpoint_path):        
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model.to(self.device)

    def preprocess_image(self, image_pil):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image.to(self.device)

    def plot_boxes(self, image_pil, boxes, labels):
        W, H = image_pil.size
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()

        for box, label in zip(boxes, labels):            
            # box = box.to(self.device)
            box = box.to("cpu")
            box = box * torch.tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.int().tolist()
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
            draw.text((x0, y0), label, fill="white", font=font)
        return image_pil

    def get_grounding_output(self, image_tensor, caption, box_thresh, text_thresh):
        image_tensor = image_tensor.to(self.device)
        caption = caption.strip().lower()
        if not caption.endswith("."):
            caption += "."

        with torch.no_grad():
            outputs = self.model(image_tensor[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        filt_mask = logits.max(dim=1)[0] > box_thresh
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        tokenized = self.model.tokenizer(caption)
        pred_phrases = [
            get_phrases_from_posmap(logit > text_thresh, tokenized, self.model.tokenizer) +
            f" ({logit.max().item():.2f})"
            for logit in logits_filt
        ]
        return boxes_filt, pred_phrases

    def crop_and_ocr(self, original_image, box):
        H, W, _ = original_image.shape
        cx, cy, bw, bh = box.tolist()

        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        cropped = original_image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped, config='--psm 7')  
        
        return cropped, text.strip()

def get_language_texts(lang):
    if lang == "Korean":
        return {
            "title": "🔍 Grounding DINO 데모",
            "description": "이미지를 업로드하고 텍스트 프롬프트에 따라 객체를 탐지해보세요.",
            "mode_label": "모드 선택",
            "modes": ["이미지", "비디오"],
            "settings": "설정",
            "config_path": "설정 파일 경로",
            "checkpoint_path": "체크포인트 파일 경로",
            "cpu_only": "CPU만 사용",
            "box_thresh": "박스 임계값",
            "text_thresh": "텍스트 임계값",
            "text_prompt": "텍스트 프롬프트",
            "default_prompt": "번호판",
            "upload_image": "또는 이미지 업로드",
            "image_dir": "이미지 폴더 경로 (선택 사항)",
            "image_preview": "### 🖼️ 랜덤 이미지 미리보기",
            "detect_button": "위 이미지 탐지하기",
            "detection_results": "### 🔍 탐지 결과",
            "original_image": "원본 이미지",
            "detected_results": "탐지된 결과",
            "ocr_result": "📝 OCR 결과",
            "cropped_region": "잘라낸 영역",
            "ocr_text": "OCR 인식 결과: ",
            "no_object": "탐지된 객체가 없습니다.",
            "video_mode": "### 🎞️ 비디오 탐지 모드",
            "select_video": "비디오 선택",
            "start_detection": "탐지 시작",
            "fps_info": "FPS: ",
            "detecting_frame": "{}초 프레임 탐지 중...",
            "results_at": "{}초 결과",
            "no_object_detected": "탐지된 객체 없음",
            "invalid_folder": "입력한 이미지 폴더 경로가 잘못되었거나 폴더가 비어있습니다.",
        }
    else:
        return {
            "title": "🔍 Grounding DINO Demo",
            "description": "Upload an image and detect objects based on your text prompt.",
            "mode_label": "Select Mode",
            "modes": ["Image", "Video"],
            "settings": "Settings",
            "config_path": "Configuration File Path",
            "checkpoint_path": "Checkpoint File Path",
            "cpu_only": "Use CPU only",
            "box_thresh": "Box Threshold",
            "text_thresh": "Text Threshold",
            "text_prompt": "Text Prompt",
            "default_prompt": "license plate",
            "upload_image": "Or upload an image",
            "image_dir": "Image Folder Path (Optional)",
            "image_preview": "### 🖼️ Random Image Preview",
            "detect_button": "Detect from Image",
            "detection_results": "### 🔍 Detection Results",
            "original_image": "Original Image",
            "detected_results": "Detected Results",
            "ocr_result": "📝 OCR Result",
            "cropped_region": "Cropped Region",
            "ocr_text": "OCR Result: ",
            "no_object": "No object detected.",
            "video_mode": "### 🎞️ Video Detection Mode",
            "select_video": "Select a video",
            "start_detection": "Start Detection",
            "fps_info": "FPS: ",
            "detecting_frame": "Detecting frame at {} sec...",
            "results_at": "Results at {} sec",
            "no_object_detected": "No object detected",
            "invalid_folder": "Invalid image folder path or folder is empty.",
        }

def process_image(config_path, checkpoint_path, cpu_only, box_thresh, text_thresh, text_prompt, uploaded_image, image_dir, lang):
    texts = get_language_texts(lang)
    device = "cpu" if cpu_only else "cuda"

    if not os.path.isfile(checkpoint_path):
        urllib.request.urlretrieve(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", f"{checkpoint_path}"
        )
        # return None, None, None, texts["invalid_folder"]

    g_dino = GroundingDINOApp(config_path=config_path, checkpoint_path=checkpoint_path, cpu_only=cpu_only, device=device)

    detection_image = None
    original_cv2 = None

    if uploaded_image is not None:
        detection_image = Image.open(uploaded_image).convert("RGB")
        original_cv2 = np.array(detection_image)
    elif os.path.isdir(image_dir):
        image_paths = glob(os.path.join(image_dir, "*.[jp][pn]g"))
        if image_paths:
            random.shuffle(image_paths)
            selected_image = image_paths[0]
            detection_image = Image.open(selected_image).convert("RGB")
            original_cv2 = np.array(detection_image)
        else:
            return None, None, None, texts["invalid_folder"]
    else:
        return None, None, None, texts["invalid_folder"]

    _, image_tensor = g_dino.preprocess_image(detection_image)
    boxes, phrases = g_dino.get_grounding_output(image_tensor, text_prompt, box_thresh, text_thresh)
    boxes = boxes.to(device)

    result_image = g_dino.plot_boxes(detection_image.copy(), boxes, phrases)

    if len(boxes) > 0:
        cropped_img, ocr_text = g_dino.crop_and_ocr(original_cv2, boxes[0])
        cleaned_text = re.sub(r'[^A-Za-z0-9\- ]', '', ocr_text)
        return detection_image, result_image, cropped_img, texts["ocr_text"] + cleaned_text
    else:
        return detection_image, result_image, None, texts["no_object"]

def process_video(config_path, checkpoint_path, cpu_only, box_thresh, text_thresh, text_prompt, video_path, lang):
    texts = get_language_texts(lang)
    device = "cpu" if cpu_only else "cuda"

    if not os.path.isfile(checkpoint_path):
        return texts["invalid_folder"]

    g_dino = GroundingDINOApp(config_path=config_path, checkpoint_path=checkpoint_path, cpu_only=cpu_only, device=device)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_frames = []

    for i in range(0, frame_count, fps):  # sample 1 frame per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        _, image_tensor = g_dino.preprocess_image(image_pil)
        boxes, phrases = g_dino.get_grounding_output(image_tensor, text_prompt, box_thresh, text_thresh)
        annotated = g_dino.plot_boxes(image_pil.copy(), boxes, phrases)
        annotated_np = np.array(annotated)
        output_frames.append(annotated_np)

    cap.release()
    return output_frames

with gr.Blocks() as demo:
    lang = gr.Radio(["English", "Korean"], label="🌐 Language / 언어 선택", value="Korean")
    texts_state = gr.State(get_language_texts("English"))

    def update_texts(lang):
        return get_language_texts(lang)

    lang.change(fn=update_texts, inputs=lang, outputs=texts_state)

    gr.Markdown("🔍 Grounding DINO Demo (Image & Video Detection)")
    with gr.Tab("📷 Image Detection"):
        with gr.Row():
            config_path = gr.Textbox(label="Config Path", value="groundingdino/config/GroundingDINO_SwinT_OGC.py")
            checkpoint_path = gr.Textbox(label="Checkpoint Path", value="groundingdino_swint_ogc.pth")
        with gr.Row():
            box_thresh = gr.Slider(0.0, 1.0, value=0.35, label="Box Threshold")
            text_thresh = gr.Slider(0.0, 1.0, value=0.25, label="Text Threshold")
            cpu_only = gr.Checkbox(label="CPU Only", value=False)
        text_prompt = gr.Textbox(label="Text Prompt", value="license plate")
        uploaded_image = gr.Image(type="filepath", label="Upload Image")
        image_dir = gr.Textbox(label="Or image folder (optional)", placeholder="lp_images")

        detect_btn = gr.Button("Run Detection")
        with gr.Row():
            original_img = gr.Image(label="Original Image")
            detected_img = gr.Image(label="Detected Results")
            cropped_img = gr.Image(label="Cropped Object")
        ocr_output = gr.Textbox(label="OCR Text Result")

        detect_btn.click(fn=process_image,
                         inputs=[config_path, checkpoint_path, cpu_only, box_thresh, text_thresh, text_prompt, uploaded_image, image_dir, lang],
                         outputs=[original_img, detected_img, cropped_img, ocr_output])

    with gr.Tab("🎞️ Video Detection"):
        video_input = gr.Video(label="Upload Video")
        run_video_btn = gr.Button("Run Video Detection")
        video_gallery = gr.Gallery(label="Detected Frames", columns=3)

        run_video_btn.click(fn=process_video,
                            inputs=[config_path, checkpoint_path, cpu_only, box_thresh, text_thresh, text_prompt, video_input, lang],
                            outputs=video_gallery)

demo.launch(share=True)