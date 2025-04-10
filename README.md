# 🔍 Grounding DINO Streamlit Demo for Image and Video Detection

This is an interactive **Streamlit-based web app** that uses **Grounding DINO** to detect objects from **images or videos** using a text prompt in either **English** or **Korean**.  
It also includes **OCR functionality** for license plate recognition.

---

## ✨ Features

- 📸 **Image Detection** with bounding boxes and OCR
- 🎥 **Video Detection** with frame-based analysis (every 30th frame)
- 🌐 **Bilingual UI**: English and Korean
- 💬 **Text-prompt-based object detection**
- 🧠 **Grounding DINO model** with selectable config and checkpoint
- 📁 **Folder-based image/video selection**
- 🔍 OCR for detected regions
- ⚡️ Deployable and sharable online using `cloudflared` or `ngrok`

---

## 🖥️ Demo Preview

| Image Mode | Video Mode |
|------------|------------|
| ![image_demo](docs/image_demo.gif) | ![video_demo](docs/video_demo.gif) |

---

## 🧰 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
