import os
import cv2
import torch
import sys
import numpy as np
import urllib.request
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from glob import glob
from time import time
import pytesseract
import random  
import re
import pandas as pd
import tempfile
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.slconfig import SLConfig

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
sys.path.append("./")

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
        
    def plot_boxes(self, image_pil, boxes, labels, scores=None, show_scores=True):
        W, H = image_pil.size
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if torch.get_device(box) != -1: box = box.to("cpu")           
            box = box * torch.tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.int().tolist()
            color = (0, 255, 0)     # Default green
            conf = 1.0
            if scores is not None:
                conf = scores[i]
                if conf > 0.6:
                    color = (0, int(conf*255), 0)  # More green with higher conf
                elif conf > 0.3:
                    color = (255, 215, 0)  # yellow
                else:
                    color = (255, 0, 0)  # Red
            draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
            txt = label
            if show_scores and scores is not None:
                txt += f" ({conf:.2f})"
            draw.text((x0, y0), txt, fill="white", font=font)
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
        scores = logits_filt.max(dim=1)[0].cpu().numpy() if logits_filt is not None else []
        tokenized = self.model.tokenizer(caption)
        pred_phrases = [
            get_phrases_from_posmap(logit > text_thresh, tokenized, self.model.tokenizer) +
            f" ({logit.max().item():.2f})"
            for logit in logits_filt
        ]
        return boxes_filt, pred_phrases, scores

    def crop_and_ocr(self, original_image, box):
        H, W, _ = original_image.shape
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * W); y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W); y2 = int((cy + bh / 2) * H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        cropped = original_image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped, config='--psm 7')  
        return cropped, text.strip()

def export_results_as(df, export_type):
    # Export to csv or Excel in memory, return file bytes/buffer
    if export_type == "CSV":
        return df.to_csv(index=False), "results.csv"
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return output, "results.xlsx"

# ------------ Streamlit App ----------------

st.set_page_config(page_title="Licence Plate Detection Streamlit Demo", layout="centered")
lang = st.sidebar.selectbox("🌐 Select Language / 언어 선택 / Tilni tanlang", ["Korean", "English", "Uzbek"])

if lang == "Korean":
    st.title("🔍 번호판 인식 프로젝트 데모")
    st.write("이미지를 업로드하고 텍스트 프롬프트에 따라 객체를 탐지해보세요.")
    mode = st.radio("모드 선택", ["이미지", "비디오", "배치", "웹캠"])
    sidebar_texts = {"config":"설정 파일 경로","checkpoint":"체크포인트 파일 경로","cpu":"CPU만 사용",
                     "box":"박스 임계값","text":"텍스트 임계값", "upload":"또는 이미지 업로드", "dir":"이미지 폴더 경로 (선택 사항)",
                     "prompt":"텍스트 프롬프트", "batch":"배치 처리", "export":"내보내기 포맷", "cam_url":"IP 카메라 RTSP/HTTP 주소"}
    labels = {"origin":"원본 이미지", "detect":"탐지된 결과", "ocr":"📝 OCR 결과", "crop":"잘라낸 영역",
              "detectbtn":"위 이미지 탐지하기", "preview":"### 🖼️ 랜덤 이미지 미리보기", "invalid_dir":"입력한 이미지 폴더 경로가 잘못되었거나 폴더가 비어있습니다.",
              "noobj":"탐지된 객체가 없습니다.", "ocr_res":"OCR 인식 결과:", "startd":"탐지 시작", "vf":"비디오 선택",
              "video":"### 🎞️ 비디오 탐지 모드", "fps":"FPS", "no_mp4":"해당 폴더에 mp4 비디오가 없습니다.",
              "progress":"탐지 중...", "down":"다운로드"}
elif lang == "Uzbek":
    st.title("🔍 Avtomobilning Davlat Raqamini Aniqlash Loyiha Demosi")
    st.write("Rasmni yuklang va matnga asoslanib obyektlarni aniqlang.")
    mode = st.radio("Rejimni tanlang:", ["Rasm", "Video", "Batch", "Webcam"])
    sidebar_texts = {"config":"Modelni sozlash fayli uchun yo'lak","checkpoint":"Model fayli uchun yo'lak","cpu":"Faqat CPU dan foydalanish",
                     "box":"Quti chegarasi","text":"Matn chegarasi", "upload":"O'z rasmingizni yuklang", "dir":"Berilgan rasmlar yo'lagi (ixtiyoriy)",
                     "prompt":"Nimani aniqlab beray?", "batch":"Batch rejimi", "export":"Saqlash turini tanlang", "cam_url":"IP/RTSP kamera (ixtiyoriy)"}
    labels = {"origin":"Original Rasm", "detect":"Natijalar", "ocr":"📝 Avtomobilning Davlat Raqami", "crop":"Davlat Raqami Rasmi",
              "detectbtn":"Ushbu Avtomobilni Tanlash", "preview":"### 🖼️ Berilgan Rasmlar", "invalid_dir":"Kiritilgan rasm papkasi yo'li noto'g'ri yoki papka bo'sh.",
              "noobj":"Avtomobilning davlat raqami aniqlanmadi.", "ocr_res":"OCR natijasi:", "startd":"Aniqlashni boshlash", "vf":"Videoni tanlang",
              "video":"### 🎞️ Video aniqlash rejimi", "fps":"FPS", "no_mp4":"Tanlangan papkada mp4 videolar topilmadi.",
              "progress":"Aniqlanmoqda...", "down":"Yuklab olish"}
else:
    st.title("🔍 Grounding DINO Demo")
    st.write("Upload an image and detect objects based on your text prompt.")
    mode = st.radio("Select Mode", ["Image", "Video", "Batch", "Webcam"])
    sidebar_texts = {"config":"Configuration File Path","checkpoint":"Checkpoint File Path","cpu":"Use CPU only",
                     "box":"Box Threshold","text":"Text Threshold", "upload":"Or upload an image", "dir":"Image Folder Path (Optional)",
                     "prompt":"Text Prompt", "batch":"Batch Mode", "export":"Export Type", "cam_url":"IP Camera RTSP/HTTP URL"}
    labels = {"origin":"Original Image", "detect":"Detection Results", "ocr":"📝 OCR Result", "crop":"Cropped Region",
              "detectbtn":"Detect from Image", "preview":"### 🖼️ Random Image Preview", "invalid_dir":"Invalid image folder path or folder is empty.",
              "noobj":"No object detected.", "ocr_res":"OCR Result:", "startd":"Start Detection", "vf":"Select a video",
              "video":"### 🎞️ Video Detection Mode", "fps":"FPS", "no_mp4":"No MP4 videos found in the selected folder.",
              "progress":"Detecting...", "down":"Download"}

with st.sidebar:
    st.header("Settings" if lang == "English" else ("설정" if lang == "Korean" else "Sozlamalar"))
    config_path = st.text_input(sidebar_texts["config"], "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    checkpoint_path = st.text_input(sidebar_texts["checkpoint"], "/home/bekhzod/Desktop/localization_models_performance/UzbekLicencePlateDetectorRecognizer/groundingdino_swint_ogc.pth")
    cpu_only = st.checkbox(sidebar_texts["cpu"], value=False)
    box_thresh = st.slider(sidebar_texts["box"], 0.0, 1.0, 0.3, 0.05)
    text_thresh = st.slider(sidebar_texts["text"], 0.0, 1.0, 0.3, 0.05)

device = "cpu" if cpu_only else "cuda"

if not os.path.isfile(checkpoint_path):    
    with st.spinner("Please wait we are downloading the pretrained weights..." if lang=="English" else (
        "잠시만 기다려 주세요. 사전 학습된 가중치를 다운로드 중입니다..." if lang=="Korean"
        else "Iltimos, kuting. Model fayllari yuklab olinmoqda..."
    )):
        urllib.request.urlretrieve(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", f"{checkpoint_path}"
        )
    st.success("Pretrained weights have been downloaded!")   

g_dino = GroundingDINOApp(config_path = config_path, checkpoint_path = checkpoint_path, cpu_only = cpu_only, device = device)    

# ========= Modes ============
def run_detection_on_one_image(image, prompt, box_thresh, text_thresh, lang, confidence_needed=True):
    pil_image = image.convert("RGB")
    original_cv2 = np.array(pil_image)
    _, image_tensor = g_dino.preprocess_image(pil_image)
    boxes, phrases, scores = g_dino.get_grounding_output(image_tensor, prompt, box_thresh, text_thresh)
    boxes = boxes.to(device)
    result_img = g_dino.plot_boxes(pil_image.copy(), boxes, phrases, scores if confidence_needed else None)
    single_results = []
    cropped_imgs = []
    ocr_texts = []
    for i, box in enumerate(boxes):
        cropped_img, ocr_text = g_dino.crop_and_ocr(original_cv2, box)
        cleaned = re.sub(r'[^A-Za-z0-9\- ]', '', ocr_text)
        
        confidence_str = "AI ishonchlilik ko'rsatkichi" if lang == "Uzbek" else "confidence"
        plate_str      = "Avtomobil Davlat Raqami" if lang == "Uzbek" else "lp_plate"
        filename_str   = "Fayl Nomi" if lang == "Uzbek" else "filename"
        single_results.append({
            f"{filename_str}":"",            
            f"{confidence_str}": (scores[i] if len(scores) > i else None),            
            f"{plate_str}":cleaned
        })
        cropped_imgs.append(cropped_img)
        ocr_texts.append(cleaned)
    return result_img, boxes, phrases, scores, cropped_imgs, ocr_texts, single_results, confidence_str, plate_str, filename_str

def st_export_csv_excel(df, lang):
    exptype = st.selectbox(sidebar_texts["export"],["CSV","Excel"])
    filebytes, fname = export_results_as(df, exptype)
    st.download_button(
        label=labels["down"],
        data=filebytes,
        file_name=fname,
        mime="text/csv" if exptype=="CSV" else "application/vnd.ms-excel"
    )

if mode == ("이미지" if lang=="Korean" else ("Image" if lang=="English" else "Rasm")):
    text_prompt = st.text_input(sidebar_texts["prompt"], "license plate" if lang!="Uzbek" else "avtomobil davlat raqami")
    if lang == "Korean" and text_prompt == "번호판":
        text_prompt = "license plate"
    if lang == "Uzbek" and text_prompt == "avtomobil davlat raqami":
        text_prompt = "license plate"

    uploaded_image = st.file_uploader(sidebar_texts["upload"], type=["png","jpg","jpeg"])
    image_dir = st.text_input(sidebar_texts["dir"], "lp_images/")
    detection_image = None
    original_cv2 = None
    detection_triggered = False

    if os.path.isdir(image_dir):
        image_paths = glob(os.path.join(image_dir, "*.[jp][pn]g"))
        random.shuffle(image_paths)
        selected_images = image_paths[:10]
        st.markdown(labels["preview"])
        rows = [selected_images[i:i+5] for i in range(0, len(selected_images), 5)]
        for row in rows:
            cols = st.columns(5)
            for col, img_path in zip(cols, row):
                with col:
                    pil_img = Image.open(img_path).convert("RGB")
                    pil_img = ImageOps.fit(pil_img, (200, 200))
                    st.image(pil_img, caption=os.path.basename(img_path), use_container_width=False)
                    if st.button(labels["detectbtn"], key=img_path):
                        detection_triggered = True
                        detection_image = Image.open(img_path).convert("RGB")
                        original_cv2 = np.array(detection_image)
                        fname = os.path.basename(img_path)
    elif image_dir.strip():
        st.warning(labels["invalid_dir"])
    if uploaded_image and not detection_triggered:
        detection_image = Image.open(uploaded_image).convert("RGB")
        original_cv2 = np.array(detection_image)
        detection_triggered = True
        fname = uploaded_image.name
    if detection_triggered and detection_image is not None:
        st.markdown("---"); st.markdown(labels["detect"])
        st.image(detection_image, caption = labels["origin"], use_container_width=True)
        with st.spinner(labels["progress"]):
            result_img, boxes, phrases, scores, cropped_imgs, ocr_texts, result_table, confidence_str, plate_str, filename_str = run_detection_on_one_image(detection_image, text_prompt, box_thresh, text_thresh, lang)
            st.image(result_img, caption=labels["detect"], use_container_width=True)

            if len(boxes) > 0:
                for i, cropped_img in enumerate(cropped_imgs):
                    st.subheader(labels["ocr"])
                    st.image(cropped_img, caption=labels["crop"], use_container_width=True)
                    st.success(f"{labels['ocr_res']} {ocr_texts[i]}")
                    buf = BytesIO()
                    img_pil = Image.fromarray(cropped_img)
                    img_pil.save(buf, format="PNG")
                    st.download_button(f"{labels['down']} {labels['crop']} #{i+1}", buf.getvalue(), file_name=f"plate_{i+1}.png", mime="image/png")

                # Export OCR results for this one image
                df = pd.DataFrame([{
                    f"{filename_str}": fname if filename_str in locals() else 'uploaded',
                    f"{confidence_str}": row[confidence_str], f"{plate_str}": row[plate_str]
                } for row in result_table])
                st_export_csv_excel(df, lang)
            else:
                st.warning(labels["noobj"])

elif mode == ("비디오" if lang=="Korean" else ("Video" if lang=="English" else "Video")):
    st.markdown(labels["video"])
    image_dir = st.text_input(sidebar_texts["dir"], "lp_images/")
    text_prompt = st.text_input(sidebar_texts["prompt"], "license plate" if lang!="Uzbek" else "avtomobil davlat raqami")
    if lang == "Uzbek" and text_prompt == "avtomobil davlat raqami":
        text_prompt = "license plate"

    video_paths = glob(os.path.join(image_dir, "*.mp4"))
    if not video_paths:
        st.warning(labels["no_mp4"])
    else:
        selected_video = st.selectbox(labels["vf"], video_paths)
        if selected_video and st.button(labels["startd"]):
            cap = cv2.VideoCapture(selected_video)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            st.info(f"{labels['fps']}: {fps}")
            frame_num = 0
            per_frame_results = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % fps == 0:  # Every 1 second
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    spinner_msg = f"{frame_num // fps + 1}초 프레임 탐지 중..." if lang == "Korean" else (
                        f"{frame_num // fps + 1}-soniya kadri aniqlanmoqda..." if lang == "Uzbek" else f"Detecting frame at {frame_num // fps + 1} sec..."
                    )
                    with st.spinner(spinner_msg):
                        _, image_tensor = g_dino.preprocess_image(pil_image)
                        boxes, phrases, scores = g_dino.get_grounding_output(image_tensor, text_prompt, box_thresh, text_thresh)
                        boxes = boxes.to(device)
                        res_caption = f"{frame_num // fps + 1}초 결과" if lang == "Korean" else (
                            f"{frame_num // fps + 1}-soniya natijasi" if lang == "Uzbek" else f"Results at {frame_num // fps + 1} sec"
                        )
                        result_image = g_dino.plot_boxes(pil_image.copy(), boxes, phrases, scores)
                        st.image(result_image, caption=res_caption, use_container_width=True)

                        per_frame = []
                        if len(boxes) > 0:
                            for i, box in enumerate(boxes):
                                cropped_img, ocr_text = g_dino.crop_and_ocr(frame, box)
                                cleaned_text = re.sub(r'[^A-Za-z0-9\- ]', '', ocr_text)
                                if lang == "Korean":
                                    st.success(f"OCR 인식 결과: {cleaned_text}")
                                elif lang == "Uzbek":
                                    st.success(f"OCR natijasi: {cleaned_text}")
                                else:
                                    st.success(f"OCR Result: {cleaned_text}")
                                buf = BytesIO()
                                Image.fromarray(cropped_img).save(buf, format="PNG")
                                st.download_button(f"{labels['down']} #{i+1} ({frame_num // fps + 1}s)", buf.getvalue(),
                                file_name=f"frame{frame_num//fps+1}_plate{i+1}.png", mime="image/png")
                                per_frame.append({
                                    "frame_s": frame_num//fps+1,
                                    "box": box.tolist(),
                                    "confidence": float(scores[i]) if len(scores)>i else None,
                                    "ocr_raw": ocr_text,
                                    "ocr_clean": cleaned_text
                                })
                        else:
                            st.info(labels["noobj"])
                        per_frame_results.extend(per_frame)
                frame_num += 1
            cap.release()
            if per_frame_results:
                df = pd.DataFrame(per_frame_results)
                st_export_csv_excel(df, lang)

elif mode.lower().startswith("batch") or mode.startswith("배치"):
    st.title("Batch Licence Plate Detection" if lang=="English" else ("배치 번호판 인식" if lang=="Korean" else "Batch Davlat Raqamini Aniqlash"))
    image_dir = st.text_input(sidebar_texts["dir"], "lp_images/")
    text_prompt = st.text_input(sidebar_texts["prompt"], "license plate" if lang!="Uzbek" else "avtomobil davlat raqami")
    if lang == "Uzbek" and text_prompt == "avtomobil davlat raqami":
        text_prompt = "license plate"
    batch_imgs = []
    results = []
    filenames = []
    if os.path.isdir(image_dir):
        image_paths = glob(os.path.join(image_dir, "*.[jp][pn]g"))
        for imgf in image_paths:
            try:
                pil = Image.open(imgf).convert("RGB")                
                result_img, boxes, phrases, scores, cropped_imgs, ocr_texts, result_table, confidence_str, plate_str, filename_str = run_detection_on_one_image(pil, text_prompt, box_thresh, text_thresh, lang)
                st.image(result_img, caption=f"{os.path.basename(imgf)} ({len(boxes)} detections)")
                for i, crimg in enumerate(cropped_imgs):
                    buf = BytesIO()
                    Image.fromarray(crimg).save(buf, format="PNG")
                    st.download_button(f"{labels['down']} {os.path.basename(imgf)} plate#{i+1}", buf.getvalue(),
                        file_name=f"batch_{os.path.basename(imgf)}_plate{i+1}.png", mime="image/png")
                    # Save results to table
                for entry in result_table:
                    entry.update({"file": os.path.basename(imgf)})
                results.extend(result_table)
            except Exception as e:
                st.warning(f"Failed on {imgf}: {e}")
        # Export table of all results
        if results:
            df = pd.DataFrame([{
                    f"{filename_str}": row[filename_str],
                    f"{confidence_str}": row[confidence_str], 
                    f"{plate_str}": row[plate_str]
                } for row in results])            
            st_export_csv_excel(df, lang)
    else:
        st.warning(labels["invalid_dir"])

elif mode.lower().startswith("webcam") or mode.lower() == "웹캠":
    import threading
    CAMERA_TYPE = st.radio("Select Camera Input", ["Webcam", "IP Camera"])
    if CAMERA_TYPE == "Webcam":
        st.write("Click Start to open webcam. Allow browser access if prompted.")
        if st.button("Start Webcam"):
            run = True
            cap = cv2.VideoCapture(0)
            FRAME_WINDOW = st.image([])
            text_prompt = st.text_input(sidebar_texts["prompt"], "license plate")
            detected_ocr = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                result_img, boxes, phrases, scores, cropped_imgs, ocr_texts, result_table, confidence_str, plate_str, filename_str = run_detection_on_one_image(pil_image, text_prompt, box_thresh, text_thresh, lang)                
                FRAME_WINDOW.image(result_img, caption="Live Webcam Detection", use_column_width=True)
                if len(ocr_texts) > 0:
                    detected_ocr = ocr_texts[0]
                if st.button("Stop Webcam"):
                    cap.release()
                    break

    else:
        cam_url = st.text_input(sidebar_texts["cam_url"], "")
        st.write("Paste RTSP/HTTP stream URL, then click Start.")
        if st.button("Start Camera"):
            if not cam_url:
                st.warning("Provide a camera URL.")
            else:
                cap = cv2.VideoCapture(cam_url)
                FRAME_WINDOW = st.image([])
                text_prompt = st.text_input(sidebar_texts["prompt"], "license plate")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    result_img, boxes, phrases, scores, cropped_imgs, ocr_texts, result_table, confidence_str, plate_str, filename_str = run_detection_on_one_image(pil_image, text_prompt, box_thresh, text_thresh, lang)                                    
                    FRAME_WINDOW.image(result_img, caption="Live IP Camera Detection", use_column_width=True)
                    if st.button("Stop Camera"):
                        cap.release()
                        break

# =================

# All main features are integrated:
# - Language support
# - Export to CSV/Excel, download detected plates
# - Webcam/IP camera detection real-time (as close as feasible in Streamlit)
# - Batch mode for entire directories
# - Visual bounding box confidence highlighting
# - Maintains compatibility with your existing workflow.
