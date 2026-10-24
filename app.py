import hashlib
import os
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import load_model, detect_image, detect_video, get_severity_label


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="RoadGuard | Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(145deg, #0a0a1a 0%, #111130 45%, #0d0d2b 100%);
        font-family: 'Inter', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f2e 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(100, 100, 255, 0.15);
    }
    .hero { text-align: center; padding: 1.8rem 1rem 1rem; }
    .hero-title {
        font-size: 2.8rem; font-weight: 800; letter-spacing: -1px;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7, #ff6ec7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-subtitle { color: #aaaacc; font-size: 1.05rem; }
    .panel {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px; padding: 1.35rem; margin-bottom: 1rem;
    }
    .section-title { color: #dedcff; font-size: 1.3rem; font-weight: 700; margin: 1.2rem 0 .7rem; }
    .metric-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem; margin: 1rem 0;
    }
    .metric {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px; padding: 1.15rem; text-align: center;
    }
    .metric-value { color: #00d4ff; font-size: 1.65rem; font-weight: 800; overflow-wrap: anywhere; }
    .metric-label { color: #aaaacc; font-size: .78rem; text-transform: uppercase; letter-spacing: .7px; }
    .muted { color: #aaaacc; }
    .stButton > button {
        background: linear-gradient(135deg, #00aeea, #7138d8);
        color: white; border: 0; border-radius: 10px; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
DEFAULT_STATE = {
    "img_result": None,
    "img_original": None,
    "img_upload_hash": None,
    "vid_result": None,
    "vid_upload_hash": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_image_result(upload_hash: str) -> None:
    """Prevent detections from a previous image being shown for a new upload."""
    if st.session_state.img_upload_hash != upload_hash:
        st.session_state.img_result = None
        st.session_state.img_original = None
        st.session_state.img_upload_hash = upload_hash


def clear_video_result(upload_hash: str) -> None:
    """Prevent video results from a previous upload being shown for a new video."""
    if st.session_state.vid_upload_hash != upload_hash:
        st.session_state.vid_result = None
        st.session_state.vid_upload_hash = upload_hash


def render_metric(value, label: str) -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        '</div>'
    )


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-title">{title}</div>'
        f'<div class="hero-subtitle">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def show_image_summary(result: dict) -> None:
    summary = result.get("summary", {})
    total = int(summary.get("total_detections", 0))
    avg_conf = float(summary.get("avg_confidence", 0.0))
    class_counts = summary.get("class_counts", {})
    severity_label, _ = get_severity_label(total)

    st.markdown(f"### Assessment: **{severity_label}**")
    st.markdown(
        '<div class="metric-grid">'
        + render_metric(total, "Total detections")
        + render_metric(f"{avg_conf:.1%}", "Average confidence")
        + render_metric(len(class_counts), "Classes detected")
        + "</div>",
        unsafe_allow_html=True,
    )

    detections = result.get("detections", [])
    if class_counts:
        st.markdown('<div class="section-title">Class summary</div>', unsafe_allow_html=True)
        class_df = pd.DataFrame(
            list(class_counts.items()), columns=["Class", "Count"]
        ).sort_values("Count", ascending=False)
        st.dataframe(class_df, use_container_width=True, hide_index=True)

    if detections:
        rows = []
        for index, detection in enumerate(detections, start=1):
            bbox = detection.get("bbox", [])
            rows.append(
                {
                    "#": index,
                    "Class": detection.get("class_name", "Unknown"),
                    "Confidence": f"{float(detection.get('confidence', 0)):.2%}",
                    "Bounding box": ", ".join(f"{float(v):.1f}" for v in bbox),
                }
            )
        with st.expander("View all detections"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif total == 0:
        st.info("No detections were returned. This does not guarantee that the road is damage-free.")


# ---------------------------------------------------------------------
# Sidebar settings and model loading
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🛣️ RoadGuard")
    st.caption("AI-powered road inspection")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Detection", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Model settings")
    model_choice = st.selectbox(
        "YOLO model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="Nano is usually faster; larger models may be slower and require more memory.",
    )
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Detections below this confidence threshold may be excluded.",
    )
    st.info(
        "Use a model trained for road damage to detect potholes and cracks reliably."
    )


@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    """Load a model once per selected model path."""
    return load_model(model_path)


try:
    with st.spinner(f"Loading {model_choice}..."):
        model = get_model(model_choice)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info("Check that the model file is available and the required packages are installed.")
    st.stop()

if model is None:
    st.error("The model loader returned no model. Check utils.py and the model path.")
    st.stop()


# ---------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------
if page == "🏠 Home":
    render_hero("RoadGuard", "AI-powered road damage detection system")
    st.markdown(
        """
        <div class="panel" style="text-align:center">
        RoadGuard analyzes uploaded road images and videos with a YOLO-based object
        detection model. It can display annotated results, confidence scores, class
        counts, and visual summaries to support road inspection.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="metric-grid">'
        + render_metric("YOLO", "Detection framework")
        + render_metric("Images", "Image analysis")
        + render_metric("Videos", "Video analysis")
        + render_metric("Charts", "Result analytics")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Key features</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔍 Object detection**")
        st.write("Display detected objects with class labels and confidence scores.")
    with c2:
        st.markdown("**🖼️ Image analysis**")
        st.write("Upload a road image and compare the original with the annotated output.")
    with c3:
        st.markdown("**🎬 Video analysis**")
        st.write("Process video frames and view or download the annotated video when generated.")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("**📊 Analytics**")
        st.write("Review class counts and confidence distributions.")
    with c5:
        st.markdown("**⚙️ Configurable**")
        st.write("Select an available model and adjust the confidence threshold.")
    with c6:
        st.markdown("**🛡️ Decision support**")
        st.write("Use results as an aid, not as a replacement for a professional inspection.")


# ---------------------------------------------------------------------
# Detection page
# ---------------------------------------------------------------------
elif page == "🔍 Detection":
    render_hero("Road Damage Detection", "Upload an image or video to analyze road conditions")
    mode = st.radio(
        "Detection mode",
        ["📸 Image detection", "🎬 Video detection"],
        horizontal=True,
    )
    st.divider()

    if mode == "📸 Image detection":
        uploaded_image = st.file_uploader(
            "Upload a road image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_uploader",
        )

        if uploaded_image is None:
            st.session_state.img_result = None
            st.session_state.img_original = None
            st.session_state.img_upload_hash = None
            st.info("Upload a JPG, JPEG, PNG, BMP, or WEBP image to begin.")
        else:
            image_bytes = uploaded_image.getvalue()
            upload_hash = hashlib.sha256(image_bytes).hexdigest()
            clear_image_result(upload_hash)

            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image_bgr is None:
                st.error("Could not read this image. Please upload a valid image file.")
            else:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                st.image(
                    image_rgb,
                    caption=f"Uploaded image: {uploaded_image.name} — {image_bgr.shape[1]} × {image_bgr.shape[0]}",
                    use_container_width=True,
                )

                if st.button("🚀 Detect road damage", use_container_width=True, key="detect_image"):
                    try:
                        with st.spinner("Running detection..."):
                            result = detect_image(model, image_bgr, confidence)
                        if not isinstance(result, dict):
                            raise ValueError("detect_image() must return a dictionary.")
                        st.session_state.img_result = result
                        st.session_state.img_original = image_rgb
                    except Exception as exc:
                        st.error(f"Image detection failed: {exc}")

                result = st.session_state.img_result
                if result is not None:
                    st.markdown('<div class="section-title">Detection results</div>', unsafe_allow_html=True)
                    annotated = result.get("annotated_image")
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Original image**")
                        st.image(st.session_state.img_original, use_container_width=True)
                    with right:
                        st.markdown("**Detected image**")
                        if annotated is not None:
                            st.image(annotated, use_container_width=True)
                        else:
                            st.warning("The detection function did not return an annotated image.")
                    show_image_summary(result)

    else:
        uploaded_video = st.file_uploader(
            "Upload a road video",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )

        if uploaded_video is None:
            st.session_state.vid_result = None
            st.session_state.vid_upload_hash = None
            st.info("Upload an MP4, AVI, MOV, or MKV video to begin.")
        else:
            video_bytes = uploaded_video.getvalue()
            upload_hash = hashlib.sha256(video_bytes).hexdigest()
            clear_video_result(upload_hash)
            st.video(video_bytes)

            if st.button("🚀 Detect damage in video", use_container_width=True, key="detect_video"):
                temp_path = None
                try:
                    suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(video_bytes)
                        temp_path = temp_file.name

                    progress = st.progress(0, text="Analyzing video frames...")

                    def update_progress(percent):
                        # Callback expects a fraction from 0.0 to 1.0.
                        fraction = max(0.0, min(float(percent), 1.0))
                        progress.progress(fraction, text=f"Analyzing... {fraction:.0%}")

                    with st.spinner("Processing video. This may take a while..."):
                        video_result = detect_video(
                            model,
                            temp_path,
                            confidence,
                            progress_callback=update_progress,
                        )
                    if not isinstance(video_result, dict):
                        raise ValueError("detect_video() must return a dictionary.")
                    st.session_state.vid_result = video_result
                    progress.progress(1.0, text="Processing complete.")
                except Exception as exc:
                    st.error(f"Video processing failed: {exc}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

            video_result = st.session_state.vid_result
            if video_result is not None:
                summary = video_result.get("overall_summary", {})
                total_frames = int(summary.get("total_frames", 0))
                total_detections = int(summary.get("total_detections", 0))
                avg_confidence = float(summary.get("avg_confidence", 0.0))
                fps = summary.get("fps", "N/A")
                average_detections = total_detections / max(total_frames, 1)
                severity, _ = get_severity_label(int(round(average_detections)))

                st.markdown(f"### Assessment: **{severity}**")
                st.markdown(
                    '<div class="metric-grid">'
                    + render_metric(total_frames, "Frames processed")
                    + render_metric(total_detections, "Total detections")
                    + render_metric(f"{avg_confidence:.1%}", "Average confidence")
                    + render_metric(fps, "Processing FPS")
                    + "</div>",
                    unsafe_allow_html=True,
                )

                class_counts = summary.get("class_counts", {})
                if class_counts:
                    st.markdown('<div class="section-title">Video class summary</div>', unsafe_allow_html=True)
                    video_df = pd.DataFrame(
                        list(class_counts.items()), columns=["Class", "Count"]
                    ).sort_values("Count", ascending=False)
                    st.dataframe(video_df, use_container_width=True, hide_index=True)

                output_path = video_result.get("output_path")
                if output_path and os.path.isfile(output_path):
                    st.markdown('<div class="section-title">Annotated video</div>', unsafe_allow_html=True)
                    st.video(output_path)
                    with open(output_path, "rb") as output_file:
                        st.download_button(
                            "⬇️ Download annotated video",
                            data=output_file.read(),
                            file_name="roadguard_annotated.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )
                else:
                    st.warning(
                        "Video analysis completed, but no accessible annotated output video was found. "
                        "Check the output_path returned by detect_video() in utils.py."
                    )


# ---------------------------------------------------------------------
# Analytics page
# ---------------------------------------------------------------------
elif page == "📊 Analytics":
    render_hero("Analytics Dashboard", "Visual summaries of the latest detection results")
    has_image = st.session_state.img_result is not None
    has_video = st.session_state.vid_result is not None

    if not has_image and not has_video:
        st.info("No analysis results are available yet. Run image or video detection first.")
    else:
        if has_image:
            result = st.session_state.img_result
            summary = result.get("summary", {})
            st.markdown('<div class="section-title">Image analytics</div>', unsafe_allow_html=True)
            class_counts = summary.get("class_counts", {})
            detections = result.get("detections", [])

            if class_counts:
                left, right = st.columns(2)
                with left:
                    st.markdown("**Detections by class**")
                    class_df = pd.DataFrame(
                        list(class_counts.items()), columns=["Class", "Count"]
                    ).sort_values("Count", ascending=False)
                    st.bar_chart(class_df.set_index("Class"))
                with right:
                    st.markdown("**Confidence distribution**")
                    confidences = [
                        float(item["confidence"])
                        for item in detections
                        if "confidence" in item
                    ]
                    if confidences:
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.hist(confidences, bins=15, edgecolor="white", alpha=0.8)
                        ax.set_xlabel("Confidence")
                        ax.set_ylabel("Number of detections")
                        ax.set_title("Image confidence scores")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("No confidence scores are available.")
            else:
                st.info("No detections were found in the latest image result.")

        if has_video:
            video_result = st.session_state.vid_result
            summary = video_result.get("overall_summary", {})
            st.markdown('<div class="section-title">Video analytics</div>', unsafe_allow_html=True)
            class_counts = summary.get("class_counts", {})
            if class_counts:
                class_df = pd.DataFrame(
                    list(class_counts.items()), columns=["Class", "Count"]
                ).sort_values("Count", ascending=False)
                st.bar_chart(class_df.set_index("Class"))
            else:
                st.info("No video class counts are available.")

            all_detections = video_result.get("all_detections", [])
            frame_counts = {}
            for detection in all_detections:
                if "frame" in detection:
                    frame = int(detection["frame"])
                    frame_counts[frame] = frame_counts.get(frame, 0) + 1

            if frame_counts:
                timeline = pd.DataFrame(
                    sorted(frame_counts.items()), columns=["Frame", "Detections"]
                ).set_index("Frame")
                st.markdown("**Detections over time**")
                st.line_chart(timeline)
            else:
                st.info("No per-frame detection data is available for the timeline.")


# ---------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------
elif page == "ℹ️ About":
    render_hero("About RoadGuard", "Project overview, model notes, and limitations")
    st.markdown(
        """
        <div class="panel">
        <h3>🛣️ Project overview</h3>
        RoadGuard is a demonstration application for analyzing road images and videos
        with an object-detection model. The application presents detections, confidence
        values, class summaries, and optional annotated video output.
        </div>
        <div class="panel">
        <h3>🧠 Model details</h3>
        This application uses a YOLO model loaded through <code>utils.py</code>.
        The model's actual capabilities depend on its training data and class labels.
        A standard pretrained YOLOv8 model is not automatically a road-damage detector.
        </div>
        <div class="panel">
        <h3>⚠️ Important limitation</h3>
        This project is for educational and decision-support purposes only. Do not use
        its output as the sole basis for road safety or maintenance decisions. Confirm
        findings through qualified human inspection.
        </div>
        <div class="panel">
        <h3>🧰 Built with</h3>
        Python · Streamlit · Ultralytics YOLO · OpenCV · NumPy · Pandas · Matplotlib
        </div>
        """,
        unsafe_allow_html=True,
    )
