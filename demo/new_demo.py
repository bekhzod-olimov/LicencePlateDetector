import os
import cv2
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# or simply:
# torch.classes.__path__ = []
import sys
sys.path.append("./")
import gc
import numpy as np
import urllib.request
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from time import time
import pytesseract
import random  
import re
from PIL import ImageOps

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
            if torch.get_device(box) != -1: print("Moving boxes to cpu..."); box = box.to("cpu")           
            
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
    
# Language selection
st.set_page_config(page_title="Streamlit Demo", layout="centered")
lang = st.sidebar.selectbox("🌐 Select Language / 언어 선택", ["English", "Korean"])

# Language-specific text
if lang == "Korean":
    st.title("🔍 차량 번호판 인식 데모")
    st.write("이미지를 업로드하고 텍스트 프롬프트에 따라 객체를 탐지해보세요.")
    mode = st.radio("모드 선택", ["이미지", "비디오"])

    with st.sidebar:
        st.header("설정")
        config_path = st.text_input("설정 파일 경로", "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        checkpoint_path = st.text_input("체크포인트 파일 경로", "groundingdino_swint_ogc.pth")
        cpu_only = st.checkbox("CPU만 사용", value=True)
        box_thresh = st.slider("박스 임계값", 0.0, 1.0, 0.3, 0.05)
        text_thresh = st.slider("텍스트 임계값", 0.0, 1.0, 0.3, 0.05)

    
    st.title("🧠 차량 번호판 인식 앱")
    text_prompt = st.text_input("텍스트 프롬프트", "번호판")
    if text_prompt == "번호판": text_prompt = "license plate"

    uploaded_image = st.file_uploader("또는 이미지 업로드", type=["png", "jpg", "jpeg"])
    image_dir = st.text_input("이미지 폴더 경로 (선택 사항)", "lp_images/")

else:  # English interface
    st.title("🔍 Licence Plate Detector Demo")
    st.write("Upload an image and detect objects based on your text prompt.")
    mode = st.radio("Select Mode", ["Image", "Video"])

    with st.sidebar:
        st.header("Settings")
        config_path = st.text_input("Configuration File Path", "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        checkpoint_path = st.text_input("Checkpoint File Path", "groundingdino_swint_ogc.pth")
        cpu_only = st.checkbox("Use CPU only", value=True)
        box_thresh = st.slider("Box Threshold", 0.0, 1.0, 0.3, 0.05)
        text_thresh = st.slider("Text Threshold", 0.0, 1.0, 0.3, 0.05)

    st.title("🧠 Vehicle License Plate Recognition App")
    text_prompt = st.text_input("Text Prompt", "license plate")

    uploaded_image = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
    image_dir = st.text_input("Image Folder Path (Optional)", "lp_images/")

# Initialize model
# device = "cpu" if cpu_only else "cuda"
device = "cpu"
# print(f"device -> {device}")

if not os.path.isfile(checkpoint_path):    
    with st.spinner("Please wait we are downloading the pretrained weights..."):
        urllib.request.urlretrieve(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", f"{checkpoint_path}"
        )
    st.success("Pretrained weights have been downloaded!")    



@st.cache_resource
def load_model(): return GroundingDINOApp(config_path=config_path, checkpoint_path=checkpoint_path, cpu_only=cpu_only, device=device)

g_dino = load_model()

@st.cache_data
def get_random_images(image_dir, sample_size=10):
    image_paths = glob(os.path.join(image_dir, "*.[jp][pn]g"))
    random.shuffle(image_paths)
    return image_paths[:sample_size]

detection_triggered = False
detection_image = None
original_cv2 = None
ocr_text = ""
preview_cache = {}

if mode == ("이미지" if lang == "Korean" else "Image"):
    if os.path.isdir(image_dir):
        selected_images = get_random_images(image_dir)

        st.markdown("### 🖼️ 랜덤 이미지 미리보기" if lang == "Korean" else "### 🖼️ Random Image Preview")
        rows = [selected_images[i:i+5] for i in range(0, len(selected_images), 5)]
        for row in rows:
            cols = st.columns(5)
            for col, img_path in zip(cols, row):
                with col:
                    if img_path not in preview_cache:
                        with Image.open(img_path) as pil_img:
                            pil_img = pil_img.convert("RGB")
                            preview_img = ImageOps.fit(pil_img, (200, 200))
                            preview_cache[img_path] = (preview_img, pil_img.copy())
                            pil_img.close()
                    preview_img, full_img = preview_cache[img_path]
                    st.image(preview_img, caption=os.path.basename(img_path), use_container_width=False)

                    if st.button("위 이미지 탐지하기" if lang == "Korean" else "Detect from Image", key=img_path):
                        detection_triggered = True
                        detection_image = full_img
                        original_cv2 = np.array(detection_image)

    elif image_dir.strip():
        st.warning("입력한 이미지 폴더 경로가 잘못되었거나 폴더가 비어있습니다." if lang == "Korean" else "Invalid image folder path or folder is empty.")

    if uploaded_image and not detection_triggered:
        detection_image = Image.open(uploaded_image).convert("RGB")
        original_cv2 = np.array(detection_image)
        detection_triggered = True

    if detection_triggered and detection_image is not None:
        st.markdown("---")
        st.markdown("### 🔍 탐지 결과" if lang == "Korean" else "### 🔍 Detection Results")
        st.image(detection_image, caption="원본 이미지" if lang == "Korean" else "Original Image", use_container_width=True)

        with st.spinner("탐지 중..." if lang == "Korean" else "Detecting..."):
            _, image_tensor = g_dino.preprocess_image(detection_image)
            boxes, phrases = g_dino.get_grounding_output(image_tensor, text_prompt, box_thresh, text_thresh)
            boxes = boxes.to(device)

            result_image = g_dino.plot_boxes(detection_image, boxes, phrases)
            st.image(result_image, caption="탐지된 결과" if lang == "Korean" else "Detected Results", use_container_width=True)

            if len(boxes) > 0:
                cropped_img, ocr_text = g_dino.crop_and_ocr(original_cv2, boxes[0])
                st.subheader("📝 OCR 결과" if lang == "Korean" else "📝 OCR Result")
                st.image(cropped_img, caption="잘라낸 영역" if lang == "Korean" else "Cropped Region", use_container_width=True)

                cleaned_text = re.sub(r'[^A-Za-z0-9\- ]', '', ocr_text[:300])
                st.success(f"OCR 인식 결과: {cleaned_text}" if lang == "Korean" else f"OCR Result: {cleaned_text}")
                del cropped_img
            else:
                st.warning("탐지된 객체가 없습니다." if lang == "Korean" else "No object detected.")
            del result_image, detection_image, original_cv2, image_tensor
            gc.collect()

else:
    st.markdown("### 🎞️ 비디오 탐지 모드" if lang == "Korean" else "### 🎞️ Video Detection Mode")

    video_paths = glob(os.path.join(image_dir, "*.mp4"))
    if not video_paths:
        st.warning("해당 폴더에 mp4 비디오가 없습니다." if lang == "Korean" else "No MP4 videos found in the selected folder.")
    else:
        selected_video = st.selectbox("비디오 선택" if lang == "Korean" else "Select a video", video_paths)
        if selected_video and st.button("탐지 시작" if lang == "Korean" else "Start Detection"):
            cap = cv2.VideoCapture(selected_video)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            st.info(f"FPS: {fps}")
            frame_num = 0
            max_frames = 30  # Limit for low memory

            while cap.isOpened() and frame_num < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % fps == 0:  # Process every second
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Resize for memory efficiency
                    max_width = 640
                    if rgb_frame.shape[1] > max_width:
                        scale = max_width / rgb_frame.shape[1]
                        rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)

                    pil_image = Image.fromarray(rgb_frame)

                    with st.spinner(f"{frame_num // fps + 1}초 프레임 탐지 중..." if lang == "Korean" else f"Detecting frame at {frame_num // fps + 1} sec..."):
                        _, image_tensor = g_dino.preprocess_image(pil_image)
                        boxes, phrases = g_dino.get_grounding_output(image_tensor, text_prompt, box_thresh, text_thresh)
                        boxes = boxes.to(device)

                        result_image = g_dino.plot_boxes(pil_image, boxes, phrases)
                        st.image(result_image, caption=f"{frame_num // fps + 1}초 결과" if lang == "Korean" else f"Results at {frame_num // fps + 1} sec", use_container_width=True)

                        if len(boxes) > 0:
                            cropped_img, ocr_text = g_dino.crop_and_ocr(frame, boxes[0])
                            cleaned_text = re.sub(r'[^A-Za-z0-9\- ]', '', ocr_text[:300])
                            st.success(f"OCR 인식 결과: {cleaned_text}" if lang == "Korean" else f"OCR Result: {cleaned_text}")
                            del cropped_img
                        else:
                            st.info("탐지된 객체 없음" if lang == "Korean" else "No object detected")
                        del result_image, pil_image, image_tensor
                        gc.collect()

                frame_num += 1
            cap.release()