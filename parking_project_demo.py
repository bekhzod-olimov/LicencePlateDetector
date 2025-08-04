import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import pytesseract
import pandas as pd
import sqlite3
import tempfile
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")

LANGS = {
    "en": {
        "title": "🚗 AI Parking Management Demo",
        "sel_lang": "Language / 언어 / Til",
        "fee_set": "Set fee per hour ($)",
        "video_src": "Video Source",
        "webcam": "Webcam",
        "ipcam": "IP Camera (RTSP/HTTP)",
        "mp4": "Video File (MP4)",
        "ip_url": "IP Camera URL",
        "start": "Start Detection",
        "entry_label": "ENTRY",
        "in_label": "(IN)",
        "exit_label": "EXIT",
        "out_label": "(OUT)",
        "stop": "Stop Detection",
        "db_table": "Parking Records",
        "download": "Download CSV",
        "plate": "License Plate",
        "entry": "Entry Time",
        "exit": "Exit Time",
        "paid": "Fee ($)",
        "no_plate": "No license plate detected.",
        "wait_cam": "Starting camera...",
        "detection": "License plate detected: ",
        "left": "Vehicle exit recorded. Total fee: $",
        "err": "Camera error or source not found.",
        "currency": "dollars",
        "foot": "Demo: AI Parking Management | "
    },
    "ko": {
        "title": "🚗 AI주차 관리 데모",
        "sel_lang": "언어 / Language / Til",
        "fee_set": "시간당 요금 설정 (₩)",
        "video_src": "비디오 소스 선택",
        "webcam": "웹캠",
        "ipcam": "IP 카메라 (RTSP/HTTP)",
        "mp4": "비디오 파일 (MP4)",
        "ip_url": "IP 카메라 주소",
        "start": "탐지 시작",
        "entry_label": "입차",  
        "in_label": "(입장)",   
        "exit_label": "출차",   
        "out_label": "(퇴장)",  
        "stop": "탐지 중지",
        "db_table": "주차 기록",
        "download": "CSV 다운로드",
        "plate": "차량 번호",
        "entry": "입장 시각",
        "exit": "퇴장 시각",
        "paid": "요금 (₩)",
        "no_plate": "번호판이 감지되지 않았습니다.",
        "wait_cam": "카메라 시작 중...",
        "detection": "탐지된 번호판: ",
        "left": "차량이 출차되었습니다. 총 요금: ₩",
        "err": "카메라 오류 또는 소스 없음.",
        "currency": "원",
        "foot": "데모: AI주차 관리 | "
    },
    "uz": {
        "title": "🚗 AI Avtoturargoh Nazoratchisi demo",
        "sel_lang": "Til / Language / 언어",
        "fee_set": "Soatbay to'lovni belgilang (so'm)",
        "video_src": "Video manbasi",
        "webcam": "Webkamera",
        "ipcam": "IP kamera (RTSP/HTTP)",
        "mp4": "Video fayl (MP4)",
        "ip_url": "IP kamera manzili",
        "start": "Aniqlashni boshlash",
        "entry_label": "KIRISH",   # Uzbek for "entered"
        "in_label": "KIRMOQDA",    # in Uzbek style
        "exit_label": "CHIQISH",   # "exited"
        "out_label": "CHIQMOQDA",  # out in Uzbek
        "stop": "To'xtatish",
        "db_table": "Avtoturargoh qaydnomasi",
        "download": "CSV yuklab olish",
        "plate": "Raqam belgisi",
        "entry": "Kirish vaqti",
        "exit": "Chiqish vaqti",
        "paid": "To'lov (so'm)",
        "no_plate": "Davlat raqami aniqlanmadi.",
        "wait_cam": "Kamera ishga tushirilmoqda...",
        "detection": "Davlat Raqami ",
        "left": "Umumiy to'lov ",
        "err": "Kamera xatosi yoki manba yo'q.",
        "currency": "so'm",
        "foot": "AI Avtoturargoh Nazoratchisi | "
    }
}

import torch
import cv2
from PIL import Image
import numpy as np
import pytesseract

from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.slconfig import SLConfig
import groundingdino.datasets.transforms as T

class GroundingDINOLicensePlateRecognizer:
    def __init__(self, config_path, checkpoint_path, device=None, box_thresh=0.15, text_thresh=0.15):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load model config and weights
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        self.model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh

    def preprocess_image(self, image_pil):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image.to(self.device)

    def get_grounding_output(self, image_tensor, caption, box_thresh=None, text_thresh=None):
        box_thresh = box_thresh if box_thresh is not None else self.box_thresh
        text_thresh = text_thresh if text_thresh is not None else self.text_thresh
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

    def crop_and_ocr(self, frame, box):
        # frame: numpy array (H, W, 3), OpenCV format
        H, W, _ = frame.shape
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        cropped = frame[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped, config='--psm 7')
        return cropped, text.strip()

    def detect_plate_with_box(self, frame, text_prompt="license plate", box_thresh=None, text_thresh=None):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, image_tensor = self.preprocess_image(pil_image)
        boxes, _, _ = self.get_grounding_output(
            image_tensor, text_prompt, box_thresh=box_thresh, text_thresh=text_thresh
        )
        boxes = boxes.to(self.device)
        for box in boxes:
            cropped_img, text = self.crop_and_ocr(frame, box)
            plate = "".join(c for c in text if c.isalnum())
            if 5 < len(plate) < 13:
                # Return both plate and box
                return plate.upper(), box
        return None, None


import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pytesseract
import pandas as pd
import sqlite3
import tempfile
from datetime import datetime
import torch
import time
import os, urllib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")

checkpoint_path = "groundingdino_swint_ogc.pth"

# Instantiate once
recognizer = GroundingDINOLicensePlateRecognizer(
    config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
    checkpoint_path=checkpoint_path
)

def init_db():
    conn = sqlite3.connect("parking.sqlite")
    conn.execute('''CREATE TABLE IF NOT EXISTS parking
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   plate TEXT,
                   entry_time TEXT,
                   exit_time TEXT,
                   fee REAL)''')
    conn.commit()
    return conn

def insert_entry(conn, plate, entry_time):
    conn.execute(
        "INSERT INTO parking (plate, entry_time, exit_time, fee) VALUES (?, ?, ?, ?)",
        (plate, entry_time, "", 0.0),
    )
    conn.commit()

def update_exit(conn, plate, exit_time, fee):
    conn.execute(
        """UPDATE parking SET exit_time=?, fee=?
           WHERE plate=? AND (exit_time='' OR exit_time IS NULL) ORDER BY entry_time DESC LIMIT 1""",
        (exit_time, fee, plate),
    )
    conn.commit()

def query_data(conn):
    return pd.read_sql_query(
        "SELECT plate, entry_time, exit_time, fee FROM parking ORDER BY entry_time DESC",
        conn,
    )

def plate_has_open_entry(conn, plate):
    """Returns True if the plate has an entry without exit_time recorded"""
    df = pd.read_sql_query(
        "SELECT 1 FROM parking WHERE plate=? AND (exit_time='' OR exit_time IS NULL) LIMIT 1",
        conn,
        params=(plate,),
    )
    return not df.empty

st.set_page_config(page_title="Parking Management Demo", layout="wide")
lang_code = st.sidebar.selectbox(
    LANGS["en"]["sel_lang"],
    ["English", "Korean", "Uzbek"],
    format_func=lambda x: {"English": "English", "Korean": "한국어", "Uzbek": "Oʻzbek"}[x],
).lower()[:2]

if not os.path.isfile(checkpoint_path):    
    with st.spinner("Please wait we are downloading the pretrained weights..." if lang_code=="English" else (
        "잠시만 기다려 주세요. 사전 학습된 가중치를 다운로드 중입니다..." if lang_code=="Korean"
        else "Iltimos, kuting. Model fayllari yuklab olinmoqda..."
    )):
        urllib.request.urlretrieve(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", f"{checkpoint_path}"
        )
    st.success("Pretrained weights have been downloaded!")

STR = LANGS[lang_code]
st.title(STR["title"])
fee_per_hour = st.sidebar.number_input(STR["fee_set"], min_value=0, value=2000 if lang_code == "uz" else 10)
video_mode = st.sidebar.selectbox(
    STR["video_src"],
    [STR["webcam"], STR["ipcam"], STR["mp4"]],
    index=2
)

if video_mode == STR["webcam"]:
    video_src = 0
elif video_mode == STR["ipcam"]:
    video_src = st.sidebar.text_input(STR["ip_url"], "")
else:
    uploaded_file = st.sidebar.file_uploader(STR["mp4"], type=["mp4"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_src = tfile.name
    else:
        video_src = None

conn = init_db()
running = st.empty()
table = st.empty()
download_div = st.empty()
stop_zone = st.empty()

if "detection_running" not in st.session_state:
    st.session_state.detection_running = False

# Debounce dictionary for IN/OUT events
if "last_event_time" not in st.session_state:
    st.session_state.last_event_time = {}
debounce_seconds = 1  # Event will be accepted only every 15s for a plate (test with small value, >0 for practical)

def start_detection():
    st.session_state.detection_running = True

def stop_detection():
    st.session_state.detection_running = False

if not st.session_state.detection_running:
    if st.button(STR["start"], key="start_button"):
        start_detection()
else:
    if stop_zone.button(STR["stop"], key="stop_button"):
        stop_detection()

if st.session_state.detection_running:
    st.info(STR["wait_cam"])
    cap = cv2.VideoCapture(video_src)
    try:
        while cap.isOpened() and st.session_state.detection_running:
            ret, frame = cap.read()
            if not ret:                
                break

            frame_disp = frame.copy()
            plate, box = recognizer.detect_plate_with_box(frame)
            now = datetime.now()

            if plate is not None and box is not None:
                H, W, _ = frame_disp.shape
                cx, cy, bw, bh = box.tolist()
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)
                last_time = st.session_state.last_event_time.get(plate)
                if last_time is None or (now - last_time).total_seconds() > debounce_seconds:
                    if not plate_has_open_entry(conn, plate):
                        # IN event
                        insert_entry(conn, plate, now.strftime("%Y-%m-%d %H:%M:%S"))
                        st.session_state.last_event_time[plate] = now

                        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame_disp,
                            f"{STR['entry_label']}: {plate}",
                            (x1, y1 - 10 if y1 > 20 else y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                        st.success(f"{STR['detection']}{plate} {STR['in_label']}...")
                    else:
                        # OUT event
                        entry_df = pd.read_sql_query(
                            "SELECT entry_time FROM parking WHERE plate=? AND (exit_time='' OR exit_time IS NULL) ORDER BY entry_time DESC LIMIT 1",
                            conn,
                            params=(plate,),
                        )
                        if entry_df.empty:
                            insert_entry(conn, plate, now.strftime("%Y-%m-%d %H:%M:%S"))
                            st.success(f"{STR['detection']}{plate} {STR['in_label']}, no entry found!")
                        else:
                            entry_time_str = entry_df.iloc[0]["entry_time"]
                            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                            exit_time = now
                            duration_min = (exit_time - entry_time).total_seconds() / 60
                            fee = round(duration_min / 60 * fee_per_hour, 2)
                            update_exit(conn, plate, exit_time.strftime("%Y-%m-%d %H:%M:%S"), fee)
                            st.session_state.last_event_time[plate] = now

                            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(
                                frame_disp,
                                f"{STR['exit_label']}: {plate}",
                                (x1, y1 - 10 if y1 > 20 else y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )
                            st.success(f"{STR['detection']}{plate} {STR['out_label']}... {STR['left']}{fee} {STR['currency']}.")

            img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            running.image(img, channels="RGB", use_container_width=True)
            df = query_data(conn)
            table.dataframe(
                df.rename(
                    columns={
                        "plate": STR["plate"],
                        "entry_time": STR["entry"],
                        "exit_time": STR["exit"],
                        "fee": STR["paid"],
                    }
                ),
                use_container_width=True,
            )
            time.sleep(1.5)
            if not st.session_state.detection_running:
                break
    finally:
        cap.release()
        running.empty()
        stop_zone.empty()
        if video_mode == STR["mp4"] and "uploaded_file" in locals() and uploaded_file:
            tfile.close()
    df = query_data(conn)
    download_div.download_button(
        label=STR["download"],
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="parking_log.csv",
        mime="text/csv",
        key="download_log_during_detection",
    )

# Show full parking table and download log button
df = query_data(conn)
st.subheader(STR["db_table"])
st.dataframe(
    df.rename(
        columns={
            "plate": STR["plate"],
            "entry_time": STR["entry"],
            "exit_time": STR["exit"],
            "fee": STR["paid"],
        }
    ),
    use_container_width=True,
)
st.download_button(
    label=STR["download"],
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="parking_log.csv",
    mime="text/csv",
    key="download_log_end",
)
st.markdown("---")
st.caption(STR["foot"] + "https://github.com/bekhzod-olimov/LicencePlateDetector")
