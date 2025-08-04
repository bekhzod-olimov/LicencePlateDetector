import streamlit as st
import cv2
from datetime import datetime
import pytesseract
import pandas as pd
import sqlite3
import tempfile
import torch
from PIL import Image
import urllib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")


# Language dictionary for multilingual support
LANGS = {
    "en": {
        "title": "🚗 AI Parking Management Demo",
        "sel_lang": "Language / 언어 / Til",
        "fee_set": "Set fee per hour ($)",
        "mp4": "Video File (MP4)",
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
        "wait_cam": "Starting video processing...",
        "detection": "License plate detected: ",
        "left": "Vehicle exit recorded. Total fee: $",
        "err": "Video source error or source not found.",
        "currency": "dollars",
        "foot": "Demo: AI Parking Management | "
    },
    "ko": {
        "title": "🚗 AI주차 관리 데모",
        "sel_lang": "언어 / Language / Til",
        "fee_set": "시간당 요금 설정 (₩)",
        "mp4": "비디오 파일 (MP4)",
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
        "wait_cam": "비디오 처리 중...",
        "detection": "탐지된 번호판: ",
        "left": "차량이 출차되었습니다. 총 요금: ₩",
        "err": "비디오 소스 오류 또는 소스 없음.",
        "currency": "원",
        "foot": "데모: AI주차 관리 | "
    },
    "uz": {
        "title": "🚗 AI Avtoturargoh Nazoratchisi demo",
        "sel_lang": "Til / Language / 언어",
        "fee_set": "Soatbay to'lovni belgilang (so'm)",
        "mp4": "Video fayl (MP4)",
        "start": "Aniqlashni boshlash",
        "entry_label": "KIRISH",
        "in_label": "KIRMOQDA",
        "exit_label": "CHIQISH",
        "out_label": "CHIQMOQDA",
        "stop": "To'xtatish",
        "db_table": "Avtoturargoh qaydnomasi",
        "download": "CSV yuklab olish",
        "plate": "Raqam belgisi",
        "entry": "Kirish vaqti",
        "exit": "Chiqish vaqti",
        "paid": "To'lov (so'm)",
        "no_plate": "Davlat raqami aniqlanmadi.",
        "wait_cam": "Video ishlanmoqda...",
        "detection": "Davlat Raqami ",
        "left": "Umumiy to'lov ",
        "err": "Video manba xatosi yoki mavjud emas.",
        "currency": "so'm",
        "foot": "AI Avtoturargoh Nazoratchisi | "
    }
}


# GroundingDINO import-related code
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.slconfig import SLConfig
import groundingdino.datasets.transforms as T


class GroundingDINOLicensePlateRecognizer:
    def __init__(self, config_path, checkpoint_path, device=None, box_thresh=0.15, text_thresh=0.15):
        self.device = "cpu"
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
        tokenized = self.model.tokenizer(caption)
        pred_phrases = [
            get_phrases_from_posmap(logit > text_thresh, tokenized, self.model.tokenizer) +
            f" ({logit.max().item():.2f})"
            for logit in logits_filt
        ]
        return boxes_filt, pred_phrases, logits_filt.max(dim=1)[0].cpu().numpy() if logits_filt is not None else []

    def crop_and_ocr(self, frame, box):
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
                return plate.upper(), box
        return None, None


# --- Database functions ---
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
    df = pd.read_sql_query(
        "SELECT 1 FROM parking WHERE plate=? AND (exit_time='' OR exit_time IS NULL) LIMIT 1",
        conn,
        params=(plate,),
    )
    return not df.empty


# --- Streamlit app setup ---
st.set_page_config(page_title="Parking Management Demo", layout="wide")
lang_code = st.sidebar.selectbox(
    LANGS["en"]["sel_lang"],
    ["English", "Korean", "Uzbek"],
    format_func=lambda x: {"English": "English", "Korean": "한국어", "Uzbek": "Oʻzbek"}[x],
).lower()[:2]
STR = LANGS[lang_code]
st.title(STR["title"])
fee_per_hour = st.sidebar.number_input(STR["fee_set"], min_value=0, value=2000 if lang_code == "uz" else 10)

checkpoint_path = "groundingdino_swint_ogc.pth"
import os
if not os.path.isfile(checkpoint_path):
    with st.spinner("Downloading pretrained weights..."):
        urllib.request.urlretrieve("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", checkpoint_path)
    st.success("Pretrained weights downloaded!")

@st.cache_resource
def load_model():
    return GroundingDINOLicensePlateRecognizer("groundingdino/config/GroundingDINO_SwinT_OGC.py", checkpoint_path)

recognizer = load_model()

st.sidebar.markdown("### Video File Input (MP4 only)")
uploaded_file = st.sidebar.file_uploader(STR["mp4"], type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_src = tfile.name
    st.session_state.video_src = video_src
else:
    video_src = st.session_state.get('video_src', None)

conn = init_db()

# Manage VideoCapture persistently in session
if "cap" not in st.session_state:
    st.session_state.cap = None

def open_video(src):
    return cv2.VideoCapture(src) if src else None

def release_video():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

if "detection_running" not in st.session_state:
    st.session_state.detection_running = False

if "last_event_time" not in st.session_state:
    st.session_state.last_event_time = {}

# Start / Stop buttons
if not st.session_state.detection_running:
    if st.button(STR["start"]):
        if not video_src:
            st.warning("Upload a video file to start detection.")
        else:
            st.session_state.cap = open_video(video_src)
            st.session_state.detection_running = True
else:
    if st.button(STR["stop"]):
        st.session_state.detection_running = False
        release_video()

# Process one frame per rerun
if st.session_state.detection_running and st.session_state.cap is not None:
    st.info(STR["wait_cam"])

    ret, frame = st.session_state.cap.read()
    if not ret:
        st.success("Video processing completed.")
        release_video()
        st.session_state.detection_running = False
    else:
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
            debounce_seconds = 1
            if last_time is None or (now - last_time).total_seconds() > debounce_seconds:
                if not plate_has_open_entry(conn, plate):
                    insert_entry(conn, plate, now.strftime("%Y-%m-%d %H:%M:%S"))
                    st.session_state.last_event_time[plate] = now
                
                    cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_disp, f"{STR['entry_label']}: {plate}",
                                (x1, y1 - 10 if y1 > 20 else y2 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    st.success(f"{STR['detection']}{plate} {STR['in_label']}...")
                else:
                    entry_df = pd.read_sql_query(
                        "SELECT entry_time FROM parking WHERE plate=? AND (exit_time='' OR exit_time IS NULL) ORDER BY entry_time DESC LIMIT 1",
                        conn, params=(plate,))
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
                        cv2.putText(frame_disp, f"{STR['exit_label']}: {plate}",
                                    (x1, y1 - 10 if y1 > 20 else y2 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        st.success(f"{STR['detection']}{plate} {STR['out_label']}... {STR['left']}{fee} {STR['currency']}.")

        img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
        st.image(img, channels="RGB", use_container_width=True)

        df = query_data(conn)
        st.dataframe(df.rename(columns={
            "plate": STR["plate"],
            "entry_time": STR["entry"],
            "exit_time": STR["exit"],
            "fee": STR["paid"],
        }), use_container_width=True)

        # Short delay, then rerun to process next frame
        import time
        time.sleep(0.1)
        st.rerun()
else:
    st.info("Please upload a video file and start detection.")

# Display and download parking records
conn = init_db()
df = query_data(conn)
st.subheader(STR["db_table"])
st.dataframe(df.rename(columns={
    "plate": STR["plate"],
    "entry_time": STR["entry"],
    "exit_time": STR["exit"],
    "fee": STR["paid"],
}), use_container_width=True)
st.download_button(label=STR["download"], data=df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="parking_log.csv", mime="text/csv", key="download_log_end")
st.markdown("---")
st.caption(STR["foot"] + "https://github.com/bekhzod-olimov/LicencePlateDetector")