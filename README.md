# 🛣️ RoadGuard - Road Damage Detection System

AI-powered road damage detection using **YOLOv8** deep learning and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)

## 🚀 Features

- **🔍 Image Detection** — Upload road images and detect damage with bounding boxes
- **🎬 Video Detection** — Process dashcam/survey videos frame-by-frame
- **📊 Analytics Dashboard** — Charts for class distribution, confidence scores, severity
- **⚙️ Configurable** — Choose model size (nano/small/medium) and confidence threshold
- **🛡️ Severity Rating** — Automatic road condition assessment

## 📸 Screenshots

| Home Page | Detection | Analytics |
|-----------|-----------|-----------|
| Hero + Features | Upload & Detect | Charts & Graphs |

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/road-damage-detection.git
cd road-damage-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── utils.py               # YOLO detection utilities
├── download_data.py       # Sample image downloader
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── .gitignore
└── README.md
```

## 🤖 Model

Uses **YOLOv8** by [Ultralytics](https://ultralytics.com/). The model is automatically downloaded on first run.

### Road Damage Classes (when fine-tuned on RDD2022)
| Code | Type |
|------|------|
| D00 | Longitudinal Crack |
| D10 | Transverse Crack |
| D20 | Alligator Crack |
| D40 | Pothole |

## ⚠️ Disclaimer

This application is for **educational purposes only**. Do not use as the sole basis for road maintenance decisions.

## 🙏 Credits

Built with [Streamlit](https://streamlit.io/) • [YOLOv8](https://ultralytics.com/) • [OpenCV](https://opencv.org/) • [Matplotlib](https://matplotlib.org/)
