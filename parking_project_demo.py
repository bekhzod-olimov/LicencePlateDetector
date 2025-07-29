# import streamlit as st
# import cv2
# import numpy as np
# from datetime import datetime
# import pytesseract
# import pandas as pd
# import sqlite3
# import tempfile
# from collections import defaultdict
# import time

# LANGS = {
#     "en": {
#         "title": "🚗 Parking Management Demo",
#         "sel_lang": "Language / 언어 / Til",
#         "fee_set": "Set fee per hour ($)",
#         "video_src": "Video Source",
#         "webcam": "Webcam",
#         "ipcam": "IP Camera (RTSP/HTTP)",
#         "mp4": "Video File (MP4)",
#         "ip_url": "IP Camera URL",
#         "start": "Start Detection",
#         "stop": "Stop Detection",
#         "db_table": "Parking Records",
#         "download": "Download CSV",
#         "plate": "License Plate",
#         "entry": "Entry Time",
#         "exit": "Exit Time",
#         "paid": "Fee ($)",
#         "no_plate": "No license plate detected.",
#         "wait_cam": "Starting camera...",
#         "detection": "License plate detected: ",
#         "left": "Vehicle exit recorded. Total fee: $",
#         "err": "Camera error or source not found.",
#         "foot": "Demo: Streamlit Parking Management (3 Languages) | "
#     },
#     "ko": {
#         "title": "🚗 주차 관리 데모",
#         "sel_lang": "언어 / Language / Til",
#         "fee_set": "시간당 요금 설정 (₩)",
#         "video_src": "비디오 소스 선택",
#         "webcam": "웹캠",
#         "ipcam": "IP 카메라 (RTSP/HTTP)",
#         "mp4": "비디오 파일 (MP4)",
#         "ip_url": "IP 카메라 주소",
#         "start": "탐지 시작",
#         "stop": "탐지 중지",
#         "db_table": "주차 기록",
#         "download": "CSV 다운로드",
#         "plate": "차량 번호",
#         "entry": "입장 시각",
#         "exit": "퇴장 시각",
#         "paid": "요금 (₩)",
#         "no_plate": "번호판이 감지되지 않았습니다.",
#         "wait_cam": "카메라 시작 중...",
#         "detection": "탐지된 번호판: ",
#         "left": "차량이 출차되었습니다. 총 요금: ₩",
#         "err": "카메라 오류 또는 소스 없음.",
#         "foot": "데모: 스트림릿 주차 관리 (3개 언어) | "
#     },
#     "uz": {
#         "title": "🚗 Avtopark boshqaruvi demsi",
#         "sel_lang": "Til / Language / 언어",
#         "fee_set": "Soatbay to'lovni belgilang (so'm)",
#         "video_src": "Video manbasi",
#         "webcam": "Webkamera",
#         "ipcam": "IP kamera (RTSP/HTTP)",
#         "mp4": "Video fayl (MP4)",
#         "ip_url": "IP kamera manzili",
#         "start": "Aniqlashni boshlash",
#         "stop": "To'xtatish",
#         "db_table": "Avtopark yozuvlari",
#         "download": "CSV yuklab olish",
#         "plate": "Raqam belgisi",
#         "entry": "Kirish vaqti",
#         "exit": "Chiqish vaqti",
#         "paid": "To'lov (so'm)",
#         "no_plate": "Davlat raqami aniqlanmadi.",
#         "wait_cam": "Kamera ishga tushirilmoqda...",
#         "detection": "Aniqlangan raqam: ",
#         "left": "Mashina chiqdi. Umumiy to'lov: ",
#         "err": "Kamera xatosi yoki manba yo'q.",
#         "foot": "Demo: Streamlit avtopark boshqaruvi (3 til) | "
#     }
# }

# # ---- Plate Recognition Function: replace with your DINO/YOLO code as needed ----
# def detect_plate(frame):
#     """
#     Dummy detector: Replace this function with your GroundingDINO or YOLO integration.
#     For demo, it uses Tesseract to try to read the plate region (full frame).
#     """
#     try:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]
#         text = pytesseract.image_to_string(thresh, config='--psm 7')
#         plate = "".join(c for c in text if c.isalnum())
#         if len(plate) > 5 and len(plate) < 13:
#             return plate.upper()
#     except Exception:
#         pass
#     return None

# # ---- DB Functions ----
# def init_db():
#     conn = sqlite3.connect("parking.sqlite")
#     cursor = conn.cursor()
#     cursor.execute('''CREATE TABLE IF NOT EXISTS parking
#                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                    plate TEXT,
#                    entry_time TEXT,
#                    exit_time TEXT,
#                    fee REAL)''')
#     conn.commit()
#     return conn

# def insert_entry(conn, plate, entry_time):
#     conn.execute("INSERT INTO parking (plate, entry_time, exit_time, fee) VALUES (?, ?, ?, ?)",
#                  (plate, entry_time, "", 0.0))
#     conn.commit()

# def update_exit(conn, plate, exit_time, fee):
#     conn.execute("""UPDATE parking SET exit_time=?, fee=?
#                  WHERE plate=? AND exit_time='' ORDER BY entry_time DESC LIMIT 1""",
#                  (exit_time, fee, plate))
#     conn.commit()

# def query_data(conn):
#     return pd.read_sql_query("SELECT plate, entry_time, exit_time, fee FROM parking ORDER BY entry_time DESC", conn)

# # ---- Streamlit UI ----
# st.set_page_config(page_title="Parking Management Demo", layout="wide")
# lang_code = st.sidebar.selectbox(
#     LANGS["en"]["sel_lang"],
#     ["English", "Korean", "Uzbek"],
#     format_func=lambda x: {"English": "English", "Korean": "한국어", "Uzbek": "Oʻzbek"}[x]
# ).lower()[:2]  # -> "en", "ko", or "uz"
# STR = LANGS[lang_code]

# st.title(STR["title"])
# fee_per_hour = st.sidebar.number_input(STR["fee_set"], min_value=0, value=2000 if lang_code=="uz" else 10)
# video_mode = st.sidebar.selectbox(STR["video_src"], [STR["webcam"], STR["ipcam"], STR["mp4"]])

# # --- Video source selection ---
# if video_mode == STR["webcam"]:
#     video_src = 0
# elif video_mode == STR["ipcam"]:
#     video_src = st.sidebar.text_input(STR["ip_url"], "")
# else:
#     uploaded_file = st.sidebar.file_uploader(STR["mp4"], type=["mp4"])
#     if uploaded_file:
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#         tfile.write(uploaded_file.read())
#         video_src = tfile.name
#     else:
#         video_src = None

# conn = init_db()
# running = st.empty()
# table = st.empty()
# download_div = st.empty()

# # ---- Session state for control buttons ---
# if "detection_running" not in st.session_state:
#     st.session_state.detection_running = False

# def start_detection():
#     st.session_state.detection_running = True

# def stop_detection():
#     st.session_state.detection_running = False

# stop_zone = st.empty()

# # --- App main logic: Start/Stop buttons outside the loop ---
# if not st.session_state.detection_running:
#     if st.button(STR["start"], key="start_button"):
#         start_detection()
# else:
#     if stop_zone.button(STR["stop"], key="stop_button"):
#         stop_detection()

# if st.session_state.detection_running:
#     st.info(STR["wait_cam"])
#     cap = cv2.VideoCapture(video_src)
#     detected_log = {}
#     try:
#         # while cap.isOpened() and st.session_state.detection_running:
#         #     ret, frame = cap.read()
#         #     if not ret:
#         #         st.warning(STR["err"])
#         #         break
#         #     frame_disp = frame.copy()
#         #     plate = detect_plate(frame)
#         #     show_plate = None
#         #     if plate:
#         #         show_plate = plate
#         #         now = datetime.now()
#         #         if plate not in detected_log or detected_log[plate]["exit_time"] is not None:
#         #             insert_entry(conn, plate, now.strftime("%Y-%m-%d %H:%M:%S"))
#         #             detected_log[plate] = {"entry_time": now, "exit_time": None}
#         #             cv2.putText(frame_disp, f"ENTRY: {plate}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
#         #             st.success(f"{STR['detection']}{plate} (IN)")
#         #         else:
#         #             entry_time = detected_log[plate]["entry_time"]
#         #             exit_time = now
#         #             minutes = (exit_time - entry_time).total_seconds() / 60
#         #             fee = round(minutes/60 * fee_per_hour, 2)
#         #             update_exit(conn, plate, exit_time.strftime("%Y-%m-%d %H:%M:%S"), fee)
#         #             detected_log[plate]["exit_time"] = exit_time
#         #             cv2.putText(frame_disp, f"OUT: {plate} {fee}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
#         #             st.success(f"{STR['left']}{fee}")
#         #     img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
#         #     running.image(img, channels="RGB", use_container_width=True)
#         frame_counter = 0
#         while cap.isOpened() and st.session_state.detection_running:
#             ret, frame = cap.read()
#             if not ret:
#                 st.warning(STR["err"])
#                 break
#             frame_disp = frame.copy()
#             plate = detect_plate(frame)
#             show_plate = None
#             if plate:
#                 show_plate = plate
#                 now = datetime.now()
#                 if plate not in detected_log or detected_log[plate]["exit_time"] is not None:
#                     insert_entry(conn, plate, now.strftime("%Y-%m-%d %H:%M:%S"))
#                     detected_log[plate] = {"entry_time": now, "exit_time": None}
#                     cv2.putText(frame_disp, f"ENTRY: {plate}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
#                     st.success(f"{STR['detection']}{plate} (IN)")
#                 else:
#                     entry_time = detected_log[plate]["entry_time"]
#                     exit_time = now
#                     minutes = (exit_time - entry_time).total_seconds() / 60
#                     fee = round(minutes/60 * fee_per_hour, 2)
#                     update_exit(conn, plate, exit_time.strftime("%Y-%m-%d %H:%M:%S"), fee)
#                     detected_log[plate]["exit_time"] = exit_time
#                     cv2.putText(frame_disp, f"OUT: {plate} {fee}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
#                     st.success(f"{STR['left']}{fee}")
#             img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
#             running.image(img, channels="RGB", use_container_width=True)
#             frame_counter += 1
#             if frame_counter % 30 == 0:
#                 df = query_data(conn)
#                 table.dataframe(
#                     df.rename(columns={
#                         "plate": STR["plate"], "entry_time": STR["entry"],
#                         "exit_time": STR["exit"], "fee": STR["paid"]
#                     }), use_container_width=True)
#                 # download_div.download_button(
#                 #     label=STR["download"],
#                 #     data=df.to_csv(index=False).encode("utf-8-sig"),
#                 #     file_name="parking_log.csv",
#                 #     mime="text/csv",
#                 #     key="download_log_during_detection"
#                 # )
#             # time.sleep(1.5)
#             if not st.session_state.detection_running:
#                 break
#     finally:
#         cap.release()
#         running.empty()
#         stop_zone.empty()
#         if video_mode == STR["mp4"] and "uploaded_file" in locals() and uploaded_file:
#             tfile.close()

#     df = query_data(conn)
#     download_div.download_button(
#         label=STR["download"],
#         data=df.to_csv(index=False).encode("utf-8-sig"),
#         file_name="parking_log.csv",
#         mime="text/csv",
#         key="download_log_during_detection"
#     )


# # # ---- Show previous records ----
# df = query_data(conn)
# st.subheader(STR["db_table"])
# st.dataframe(
#     df.rename(columns={
#         "plate": STR["plate"], "entry_time": STR["entry"],
#         "exit_time": STR["exit"], "fee": STR["paid"]
#     }), use_container_width=True)
# st.download_button(
#     label=STR["download"],
#     data=df.to_csv(index=False).encode("utf-8-sig"),
#     file_name="parking_log.csv",
#     mime="text/csv",
#     key="download_log_end"
# )
# st.markdown("---")
# st.caption(STR["foot"] + "github.com/your-demo-link-here")

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import pytesseract
import pandas as pd
import sqlite3
import tempfile
from collections import defaultdict
import time


LANGS = {
    "en": {
        "title": "🚗 Parking Management Demo",
        "sel_lang": "Language / 언어 / Til",
        "fee_set": "Set fee per hour ($)",
        "video_src": "Video Source",
        "webcam": "Webcam",
        "ipcam": "IP Camera (RTSP/HTTP)",
        "mp4": "Video File (MP4)",
        "ip_url": "IP Camera URL",
        "start": "Start Detection",
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
        "foot": "Demo: Streamlit Parking Management (3 Languages) | "
    },
    "ko": {
        "title": "🚗 주차 관리 데모",
        "sel_lang": "언어 / Language / Til",
        "fee_set": "시간당 요금 설정 (₩)",
        "video_src": "비디오 소스 선택",
        "webcam": "웹캠",
        "ipcam": "IP 카메라 (RTSP/HTTP)",
        "mp4": "비디오 파일 (MP4)",
        "ip_url": "IP 카메라 주소",
        "start": "탐지 시작",
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
        "foot": "데모: 스트림릿 주차 관리 (3개 언어) | "
    },
    "uz": {
        "title": "🚗 Avtopark boshqaruvi demsi",
        "sel_lang": "Til / Language / 언어",
        "fee_set": "Soatbay to'lovni belgilang (so'm)",
        "video_src": "Video manbasi",
        "webcam": "Webkamera",
        "ipcam": "IP kamera (RTSP/HTTP)",
        "mp4": "Video fayl (MP4)",
        "ip_url": "IP kamera manzili",
        "start": "Aniqlashni boshlash",
        "stop": "To'xtatish",
        "db_table": "Avtopark yozuvlari",
        "download": "CSV yuklab olish",
        "plate": "Raqam belgisi",
        "entry": "Kirish vaqti",
        "exit": "Chiqish vaqti",
        "paid": "To'lov (so'm)",
        "no_plate": "Davlat raqami aniqlanmadi.",
        "wait_cam": "Kamera ishga tushirilmoqda...",
        "detection": "Aniqlangan raqam: ",
        "left": "Mashina chiqdi. Umumiy to'lov: ",
        "err": "Kamera xatosi yoki manba yo'q.",
        "foot": "Demo: Streamlit avtopark boshqaruvi (3 til) | "
    }
}


def detect_plate(frame):
    """
    Dummy detector: Replace this function with your GroundingDINO or YOLO integration.
    For demo, it uses Tesseract OCR on the whole frame.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]
        text = pytesseract.image_to_string(thresh, config="--psm 7")
        plate = "".join(c for c in text if c.isalnum())
        if len(plate) > 5 and len(plate) < 13:
            return plate.upper()
    except Exception:
        pass
    return None


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
        (
            plate,
            entry_time,
            "",
            0.0,
        ),
    )
    conn.commit()


def update_exit(conn, plate, exit_time, fee):
    conn.execute(
        """UPDATE parking SET exit_time=?, fee=?
           WHERE plate=? AND exit_time='' ORDER BY entry_time DESC LIMIT 1""",
        (exit_time, fee, plate),
    )
    conn.commit()


def query_data(conn):
    return pd.read_sql_query(
        "SELECT plate, entry_time, exit_time, fee FROM parking ORDER BY entry_time DESC",
        conn,
    )


st.set_page_config(page_title="Parking Management Demo", layout="wide")
lang_code = st.sidebar.selectbox(
    LANGS["en"]["sel_lang"],
    ["English", "Korean", "Uzbek"],
    format_func=lambda x: {"English": "English", "Korean": "한국어", "Uzbek": "Oʻzbek"}[x],
).lower()[:2]
STR = LANGS[lang_code]

st.title(STR["title"])
fee_per_hour = st.sidebar.number_input(STR["fee_set"], min_value=0, value=2000 if lang_code == "uz" else 10)
video_mode = st.sidebar.selectbox(STR["video_src"], [STR["webcam"], STR["ipcam"], STR["mp4"]])

# Video source handling
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


def start_detection():
    st.session_state.detection_running = True


def stop_detection():
    st.session_state.detection_running = False


# Controls outside frame processing loop
if not st.session_state.detection_running:
    if st.button(STR["start"], key="start_button"):
        start_detection()
else:
    if stop_zone.button(STR["stop"], key="stop_button"):
        stop_detection()


# Start detection and process frames
if st.session_state.detection_running:
    st.info(STR["wait_cam"])
    cap = cv2.VideoCapture(video_src)
    detected_log = {}

    try:        
        while cap.isOpened() and st.session_state.detection_running:
            ret, frame = cap.read()
            if not ret:
                st.warning(STR["err"])
                break
            frame_disp = frame.copy()
            plate = detect_plate(frame)
            if plate:
                now = datetime.now()
                if plate not in detected_log or detected_log[plate]["exit_time"] is not None:
                    # Car is outside or first detection (register IN)
                    insert_entry(conn, plate, now.strftime("%Y-%m-%d %H:%M:%S"))
                    detected_log[plate] = {"entry_time": now, "exit_time": None}
                    cv2.putText(
                        frame_disp,
                        f"ENTRY: {plate}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )
                    st.success(f"{STR['detection']}{plate} (IN)")
                else:
                    # Car is inside, register OUT
                    entry_time = detected_log[plate]["entry_time"]
                    exit_time = now
                    minutes = (exit_time - entry_time).total_seconds() / 60
                    fee = round(minutes / 60 * fee_per_hour, 2)
                    update_exit(conn, plate, exit_time.strftime("%Y-%m-%d %H:%M:%S"), fee)
                    detected_log[plate]["exit_time"] = exit_time
                    cv2.putText(
                        frame_disp,
                        f"OUT: {plate} {fee}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        3,
                    )
                    st.success(f"{STR['left']}{fee}")
                    # Remove the plate to reset state (next detection is IN)
                    del detected_log[plate]

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
            if not st.session_state.detection_running:
                break
    finally:
        cap.release()
        running.empty()
        stop_zone.empty()
        if video_mode == STR["mp4"] and "uploaded_file" in locals() and uploaded_file:
            tfile.close()

    # Show download button **only once** after detection loop has stopped
    df = query_data(conn)
    download_div.download_button(
        label=STR["download"],
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="parking_log.csv",
        mime="text/csv",
        key="download_log_during_detection",
    )


# Show full records with download button outside detection
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
st.caption(STR["foot"] + "github.com/your-demo-link-here")
