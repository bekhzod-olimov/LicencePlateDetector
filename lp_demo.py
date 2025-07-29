import os
import cv2
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
# or simply:
# torch.classes.__path__ = []
import sys
sys.path.append("./")

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
st.set_page_config(page_title="Licence Plate Detection Streamlit Demo", layout="centered")
lang = st.sidebar.selectbox("🌐 Select Language / 언어 선택 / Tilni tanlang", ["Korean", "English", "Uzbek"])

# Language-specific text
if lang == "Korean":
    st.title("🔍 번호판 인식 프로젝트 데모")
    st.write("이미지를 업로드하고 텍스트 프롬프트에 따라 객체를 탐지해보세요.")
    mode = st.radio("모드 선택", ["이미지", "비디오"])

    with st.sidebar:
        st.header("설정")
        config_path = st.text_input("설정 파일 경로", "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        checkpoint_path = st.text_input("체크포인트 파일 경로", "groundingdino_swint_ogc.pth")
        cpu_only = st.checkbox("CPU만 사용", value=False)
        box_thresh = st.slider("박스 임계값", 0.0, 1.0, 0.3, 0.05)
        text_thresh = st.slider("텍스트 임계값", 0.0, 1.0, 0.3, 0.05)

    st.title("🧠 차량 번호판 인식 앱")
    text_prompt = st.text_input("텍스트 프롬프트", "번호판")
    if text_prompt == "번호판": text_prompt = "license plate"

    uploaded_image = st.file_uploader("또는 이미지 업로드", type=["png", "jpg", "jpeg"])
    image_dir = st.text_input("이미지 폴더 경로 (선택 사항)", "lp_images/")

elif lang == "Uzbek":
    st.title("🔍 Avtomobilning Davlat Raqamini Aniqlash Loyiha Demosi")
    st.write("Rasmni yuklang va matnga asoslanib obyektlarni aniqlang.")
    mode = st.radio("Rejimni tanlang:", ["Rasm", "Video"])

    with st.sidebar:
        st.header("Sozlamalar")
        config_path = st.text_input("Modelni sozlash fayli uchun yo'lak", "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        checkpoint_path = st.text_input("Model fayli uchun yo'lak", "groundingdino_swint_ogc.pth")
        cpu_only = st.checkbox("Faqat CPU dan foydalanish", value=False)
        box_thresh = st.slider("Quti chegarasi", 0.0, 1.0, 0.3, 0.05)
        text_thresh = st.slider("Matn chegarasi", 0.0, 1.0, 0.3, 0.05)

    st.title("🧠 Avtomobil davlat raqamini aniqlash dasturi")
    text_prompt = st.text_input("Nimani aniqlab beray?", "avtomobil davlat raqami")
    if text_prompt == "avtomobil davlat raqami":
        text_prompt = "license plate"

    uploaded_image = st.file_uploader("O'z rasmingizni yuklang", type=["png", "jpg", "jpeg"])
    image_dir = st.text_input("Berilgan rasmlar yo'lagi (ixtiyoriy)", "lp_images/")

else:  # English interface
    st.title("🔍 Grounding DINO Demo")
    st.write("Upload an image and detect objects based on your text prompt.")
    mode = st.radio("Select Mode", ["Image", "Video"])

    with st.sidebar:
        st.header("Settings")
        config_path = st.text_input("Configuration File Path", "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        checkpoint_path = st.text_input("Checkpoint File Path", "groundingdino_swint_ogc.pth")
        cpu_only = st.checkbox("Use CPU only", value=False)
        box_thresh = st.slider("Box Threshold", 0.0, 1.0, 0.3, 0.05)
        text_thresh = st.slider("Text Threshold", 0.0, 1.0, 0.3, 0.05)

    st.title("🧠 Vehicle License Plate Recognition App")
    text_prompt = st.text_input("Text Prompt", "license plate")

    uploaded_image = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
    image_dir = st.text_input("Image Folder Path (Optional)", "lp_images/")

# Initialize model
device = "cpu" if cpu_only else "cuda"

if not os.path.isfile(checkpoint_path):    # True 
    with st.spinner("Please wait we are downloading the pretrained weights..."):
        urllib.request.urlretrieve(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", f"{checkpoint_path}"
        )
    st.success("Pretrained weights have been downloaded!")          

g_dino = GroundingDINOApp(config_path = config_path, checkpoint_path = checkpoint_path, cpu_only = cpu_only, device = device)    

# Image preview and selection
detection_triggered = False
detection_image = None
original_cv2 = None
result_image = None
cropped_img = None
ocr_text = ""

if mode == ("이미지" if lang == "Korean" else ("Image" if lang == "English" else "Rasm")):
    if os.path.isdir(image_dir):
        image_paths = glob(os.path.join(image_dir, "*.[jp][pn]g"))
        random.shuffle(image_paths)
        selected_images = image_paths[:10]

        st.markdown("### 🖼️ 랜덤 이미지 미리보기" if lang == "Korean" else ("### 🖼️ Random Image Preview" if lang == "English" else "### 🖼️ Berilgan Rasmlar"))
        rows = [selected_images[i:i+5] for i in range(0, len(selected_images), 5)]
        for row in rows:
            cols = st.columns(5)
            for col, img_path in zip(cols, row):
                with col:
                    pil_img = Image.open(img_path).convert("RGB")
                    pil_img = ImageOps.fit(pil_img, (200, 200))
                    st.image(pil_img, caption=os.path.basename(img_path), use_container_width=False)
                    if st.button("위 이미지 탐지하기" if lang == "Korean" else ("Detect from Image" if lang == "English" else "Ushbu Avtomobilni Tanlash"), key=img_path):
                        detection_triggered = True
                        detection_image = Image.open(img_path).convert("RGB")
                        original_cv2 = np.array(detection_image)
    elif image_dir.strip():
        if lang == "Uzbek":
            st.warning("Kiritilgan rasm papkasi yo'li noto'g'ri yoki papka bo'sh.")
        elif lang == "Korean":
            st.warning("입력한 이미지 폴더 경로가 잘못되었거나 폴더가 비어있습니다.")
        else:
            st.warning("Invalid image folder path or folder is empty.")       
        

    if uploaded_image and not detection_triggered:
        detection_image = Image.open(uploaded_image).convert("RGB")
        original_cv2 = np.array(detection_image)
        detection_triggered = True

    if detection_triggered and detection_image is not None:
        st.markdown("---")        
        st.markdown("### 🔍 탐지 결과" if lang == "Korean" else ("### 🔍 Detection Results" if lang == "English" else "### 🔍 Natijalar"))
        st.image(detection_image, caption = "원본 이미지" if lang == "Korean" else ("Original Image" if lang == "English" else "Original Rasm"), use_container_width=True)

        with st.spinner("탐지 중..." if lang == "Korean" else ("Detecting..." if lang == "English" else "Aniqlanmoqda...")):
            _, image_tensor = g_dino.preprocess_image(detection_image)
            boxes, phrases = g_dino.get_grounding_output(image_tensor, text_prompt, box_thresh, text_thresh)
            boxes = boxes.to(device)

            result_image = g_dino.plot_boxes(detection_image.copy(), boxes, phrases)            
            st.image(result_image, caption = "탐지된 결과" if lang == "Korean" else ("Detection Results" if lang == "English" else "Natijalar"), use_container_width=True)
            

            if len(boxes) > 0:
                cropped_img, ocr_text = g_dino.crop_and_ocr(original_cv2, boxes[0])                
                st.subheader("📝 OCR 결과" if lang == "Korean" else ("📝 OCR Result" if lang == "English" else "📝 Avtomobilning Davlat Raqami"))                
                st.image(cropped_img, caption = "잘라낸 영역" if lang == "Korean" else ("Cropped Region" if lang == "English" else "Davlat Raqami Rasmi"), use_container_width=True)

                cleaned_text = re.sub(r'[^A-Za-z0-9\- ]', '', ocr_text)
                if lang == "Uzbek":
                    st.success(f"OCR natijasi: {cleaned_text}")
                elif lang == "Korean":
                    st.success(f"OCR 인식 결과: {cleaned_text}")
                else:
                    st.success(f"OCR Result: {cleaned_text}")
                # st.success(f"OCR 인식 결과: {cleaned_text}" if lang == "Korean" else f"OCR Result: {cleaned_text}")
            else:
                st.warning("탐지된 객체가 없습니다." if lang == "Korean" else ("No object detected." if lang == "English" else "Avtomobilning davlat raqami aniqlanmadi."))

else:  # Video mode
    # Video mode title
    if lang == "Korean":
        st.markdown("### 🎞️ 비디오 탐지 모드")
    elif lang == "Uzbek":
        st.markdown("### 🎞️ Video aniqlash rejimi")
    else:
        st.markdown("### 🎞️ Video Detection Mode")

    video_paths = glob(os.path.join(image_dir, "*.mp4"))
    # Warning if no videos found
    if not video_paths:
        if lang == "Korean":
            st.warning("해당 폴더에 mp4 비디오가 없습니다.")
        elif lang == "Uzbek":
            st.warning("Tanlangan papkada mp4 videolar topilmadi.")
        else:
            st.warning("No MP4 videos found in the selected folder.")
    else:
        # Video selector
        if lang == "Korean":
            sel_video_lbl = "비디오 선택"
            btn_lbl = "탐지 시작"
        elif lang == "Uzbek":
            sel_video_lbl = "Videoni tanlang"
            btn_lbl = "Aniqlashni boshlash"
        else:
            sel_video_lbl = "Select a video"
            btn_lbl = "Start Detection"

        selected_video = st.selectbox(sel_video_lbl, video_paths)
        if selected_video and st.button(btn_lbl):
            cap = cv2.VideoCapture(selected_video)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            st.info(f"FPS: {fps}")
            frame_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % fps == 0:  # Every 1 second
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)

                    # Spinner message for detection progress
                    if lang == "Korean":
                        spinner_msg = f"{frame_num // fps + 1}초 프레임 탐지 중..."
                    elif lang == "Uzbek":
                        spinner_msg = f"{frame_num // fps + 1}-soniya kadri aniqlanmoqda..."
                    else:
                        spinner_msg = f"Detecting frame at {frame_num // fps + 1} sec..."

                    with st.spinner(spinner_msg):
                        _, image_tensor = g_dino.preprocess_image(pil_image)
                        boxes, phrases = g_dino.get_grounding_output(image_tensor, text_prompt, box_thresh, text_thresh)
                        boxes = boxes.to(device)

                        # Image result caption
                        if lang == "Korean":
                            res_caption = f"{frame_num // fps + 1}초 결과"
                        elif lang == "Uzbek":
                            res_caption = f"{frame_num // fps + 1}-soniya natijasi"
                        else:
                            res_caption = f"Results at {frame_num // fps + 1} sec"

                        result_image = g_dino.plot_boxes(pil_image.copy(), boxes, phrases)
                        st.image(result_image, caption=res_caption, use_container_width=True)

                        if len(boxes) > 0:
                            cropped_img, ocr_text = g_dino.crop_and_ocr(frame, boxes[0])
                            cleaned_text = re.sub(r'[^A-Za-z0-9\- ]', '', ocr_text)
                            # Success message for OCR
                            if lang == "Korean":
                                st.success(f"OCR 인식 결과: {cleaned_text}")
                            elif lang == "Uzbek":
                                st.success(f"OCR natijasi: {cleaned_text}")
                            else:
                                st.success(f"OCR Result: {cleaned_text}")
                        else:
                            # Info message for "no object detected"
                            if lang == "Korean":
                                st.info("탐지된 객체 없음")
                            elif lang == "Uzbek":
                                st.info("Aniqlangan obyekt yo'q")
                            else:
                                st.info("No object detected")

                frame_num += 1

            cap.release()