
from pathlib import Path
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO


# ==================================================
# CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"


# ==================================================
# SESSION STATE
# ==================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None


# ==================================================
# MODEL
# ==================================================
@st.cache_resource
def load_model(path, modified_time):
    return YOLO(path)


def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )


# ==================================================
# BEAUTIFUL UI
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #09111f;
    --panel: #111e30;
    --border: #24364c;
    --text: #f2f7ff;
    --muted: #9eafc5;
    --accent: #51e2c2;
}

.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, #16364a 0%, transparent 38%),
        #09111f;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: #0c1626;
    border-right: 1px solid var(--border);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

p, li, label {
    color: #c2cfdf;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 42px 38px;
    border-radius: 24px;
    border: 1px solid #2d5064;
    background: linear-gradient(120deg, #142b43, #10263a 55%, #124047);
    animation: enter 0.8s ease both;
}

.hero::after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -65px;
    top: -85px;
    border: 1px solid #4de4c344;
    border-radius: 50%;
    box-shadow:
        0 0 0 25px #4de4c310,
        0 0 0 50px #4de4c308;
    animation: float 7s ease-in-out infinite;
    pointer-events: none;
}

.eyebrow {
    color: var(--accent);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(34px, 5vw, 54px);
    line-height: 1.12;
    margin: 18px 0;
    font-weight: 800;
}

.hero p {
    color: #c0d1e2;
    font-size: 16px;
    line-height: 1.8;
    max-width: 660px;
}

.panel {
    background: linear-gradient(145deg, #142338, #101b2b);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 23px;
    min-height: 175px;
    animation: enter 0.7s ease both;
    transition: transform 0.25s ease, border-color 0.25s ease;
}

.panel:hover {
    transform: translateY(-4px);
    border-color: #51e2c2;
}

.panel h3 {
    font-size: 19px;
    margin-bottom: 10px;
}

.panel p {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.8;
}

.section-label {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    color: var(--accent);
    text-transform: uppercase;
    margin: 30px 0 15px;
}

div[data-testid="stMetric"] {
    background: #111e30;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #aabbd0;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
    border: 1px solid #3bc8aa;
    background: linear-gradient(110deg, #087e77, #126d83);
    color: white;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #75ffe1;
    transform: translateY(-2px);
    color: white;
}

div[data-testid="stFileUploader"] {
    border: 1px dashed #3d6277;
    border-radius: 15px;
    padding: 10px;
    background: #101d2d;
}

hr {
    border-color: #26384d;
}

.status {
    display: inline-block;
    border: 1px solid #28796b;
    background: #123c37;
    color: #70f1d1;
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 12px;
    font-weight: 700;
}

@keyframes enter {
    from { opacity: 0; transform: translateY(13px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes float {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(12px) rotate(5deg); }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## 🛣️ RoadGuard")
    st.caption("INTELLIGENT ROAD INSPECTION")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Home", "Detect Damage", "Analytics & About"],
        key="page",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**MODEL STATUS**")

    if MODEL_PATH.exists():
        st.success("best.pt available")
    else:
        st.error("best.pt missing")

    st.caption("Computer vision · YOLO")


# ==================================================
# HOME PAGE
# ==================================================
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">AI-powered road inspection</div>
        <h1>Smarter roads.<br>Safer journeys.</h1>
        <p>
            Detect potholes and road surface defects using computer vision.
            Upload a road image, inspect AI predictions, and review
            your analysis results in one simple workspace.
        </p>
        <span class="status">● ROAD DAMAGE DETECTION</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">Your workspace</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="panel">
            <h3>📷 Image Detection</h3>
            <p>Upload a road image and view detected defects with
            bounding boxes and confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="panel">
            <h3>⚡ Custom YOLO Model</h3>
            <p>Run predictions using your own trained best.pt
            weights instead of a generic object detector.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="panel">
            <h3>📊 Result Analytics</h3>
            <p>Review detection counts, confidence levels and
            recent image analysis from your current session.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">Session overview</div>',
        unsafe_allow_html=True,
    )

    total_images = len(st.session_state.history)
    total_boxes = sum(
        item["detections"] for item in st.session_state.history
    )

    m1, m2 = st.columns(2)
    m1.metric("Images analyzed", total_images)
    m2.metric("Total detections", total_boxes)

    st.write("")
    if st.button("Start detecting road damage →"):
        st.session_state.page = "Detect Damage"
        st.rerun()


# ==================================================
# DETECTION PAGE
# ==================================================
elif page == "Detect Damage":

    st.title("Road Damage Detection")
    st.write(
        "Upload an image, choose your confidence threshold, "
        "and run your custom-trained model."
    )

    uploaded = st.file_uploader(
        "Upload road image",
        type=["jpg", "jpeg", "png"],
    )

    left, right = st.columns(2)

    with left:
        confidence = st.slider(
            "Confidence threshold",
            0.10, 0.90, 0.25, 0.05,
        )

    with right:
        image_size = st.select_slider(
            "Inference image size",
            options=[320, 416, 512, 640],
            value=416,
        )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(
            image,
            caption="Selected road image",
            use_container_width=True,
        )

        if st.button("Analyze image", use_container_width=True):
            try:
                if not MODEL_PATH.exists():
                    st.error(
                        f"Place best.pt beside app.py. Expected: {MODEL_PATH}"
                    )
                    st.stop()

                with st.spinner("Analyzing road surface..."):
                    model = get_model()
                    result = model.predict(
                        source=image,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False,
                    )[0]

                boxes = result.boxes
                count = len(boxes) if boxes is not None else 0

                # Convert Ultralytics BGR output to RGB.
                annotated = result.plot()[:, :, ::-1].copy()

                buffer = BytesIO()
                Image.fromarray(annotated).save(buffer, format="PNG")

                class_counts = {}
                scores = []

                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        score = float(box.conf[0].item())
                        scores.append(score)

                        if isinstance(model.names, dict):
                            label = model.names.get(
                                class_id, f"Class {class_id}"
                            )
                        else:
                            label = model.names[class_id]

                        class_counts[label] = (
                            class_counts.get(label, 0) + 1
                        )

                mean_conf = (
                    sum(scores) / len(scores) if scores else None
                )

                st.session_state.result = {
                    "original": image.copy(),
                    "annotated": annotated,
                    "download": buffer.getvalue(),
                    "count": count,
                    "mean_conf": mean_conf,
                    "classes": class_counts,
                    "filename": uploaded.name,
                }

                st.session_state.history.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "image": uploaded.name,
                    "detections": count,
                    "confidence": mean_conf,
                    "classes": class_counts.copy(),
                })

            except Exception as e:
                st.error("Detection failed.")
                st.exception(e)

    output = st.session_state.result

    if output is not None:
        st.divider()
        st.subheader("Detection results")

        a, b = st.columns(2)

        with a:
            st.markdown("**Original**")
            st.image(
                output["original"],
                use_container_width=True,
            )

        with b:
            st.markdown("**AI prediction**")
            st.image(
                output["annotated"],
                use_container_width=True,
            )

        k1, k2 = st.columns(2)
        k1.metric("Detected objects", output["count"])
        k2.metric(
            "Mean confidence",
            (
                f'{output["mean_conf"]:.1%}'
                if output["mean_conf"] is not None
                else "N/A"
            ),
        )

        if output["count"] == 0:
            st.warning(
                "No detections found. Try another image or lower "
                "the confidence threshold."
            )
        else:
            st.markdown("**Detected classes**")
            st.json(output["classes"])

        st.download_button(
            "Download annotated image",
            data=output["download"],
            file_name="roadguard_result.png",
            mime="image/png",
            use_container_width=True,
        )


# ==================================================
# ANALYTICS & ABOUT PAGE
# ==================================================
elif page == "Analytics & About":

    st.title("Analytics & About")
    st.write("A summary of the predictions made in this session.")

    history = st.session_state.history

    if history:
        total_images = len(history)
        total_boxes = sum(
            item["detections"] for item in history
        )

        scores = [
            item["confidence"]
            for item in history
            if item["confidence"] is not None
        ]

        mean_score = sum(scores) / len(scores) if scores else None

        a, b, c = st.columns(3)
        a.metric("Images analyzed", total_images)
        b.metric("Total detections", total_boxes)
        c.metric(
            "Mean confidence",
            f"{mean_score:.1%}" if mean_score is not None else "N/A",
        )

        st.subheader("Detections per image")

        chart_df = pd.DataFrame([
            {
                "Image": item["image"],
                "Detections": item["detections"],
            }
            for item in history
        ])

        st.bar_chart(
            chart_df.set_index("Image"),
            y="Detections",
        )

        all_classes = {}

        for item in history:
            for label, count in item["classes"].items():
                all_classes[label] = all_classes.get(label, 0) + count

        if all_classes:
            st.subheader("Detections by class")
            class_df = pd.DataFrame(
                list(all_classes.items()),
                columns=["Class", "Count"],
            ).set_index("Class")

            st.bar_chart(class_df)

        st.subheader("Recent analysis")
        st.dataframe(
            pd.DataFrame([
                {
                    "Time": item["time"],
                    "Image": item["image"],
                    "Detections": item["detections"],
                    "Confidence": (
                        f'{item["confidence"]:.1%}'
                        if item["confidence"] is not None
                        else "N/A"
                    ),
                }
                for item in reversed(history)
            ]),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()

    else:
        st.info(
            "No results yet. Visit Detect Damage and analyze an image "
            "to populate these charts."
        )

    st.divider()
    st.subheader("About RoadGuard")

    st.markdown("""
    RoadGuard is a computer vision application that uses a
    custom-trained YOLO model to detect road surface defects.

    **Technology stack**
    - Python and Streamlit
    - Ultralytics YOLO
    - Pillow and Pandas
    - Custom-trained `best.pt` weights

    **Important:** Prediction quality depends on the training dataset
    and model validation. The interface does not guarantee that every
    pothole will be detected correctly.
    """)

    if MODEL_PATH.exists():
        try:
            model = get_model()
            st.success("Custom model loaded")
            st.write("Model classes:")
            st.json(model.names)
        except Exception as e:
            st.error("Model could not be loaded.")
            st.code(str(e))
    else:
        st.warning("best.pt is missing from the project folder.")


# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("ROADGUARD AI  ·  Intelligent road inspection")



# ---------------- LOAD CUSTOM MODEL ----------------
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

print("Model path:", MODEL_PATH)
print("Model file exists:", MODEL_PATH.is_file())




# ---------------- APP HEADER ----------------
st.title("🛣️ RoadGuard")
st.subheader("AI-Powered Road Damage & Pothole Detection")
st.write(
    "Upload a road image to detect potholes and other road defects "
    "using your custom-trained YOLO model."
)

st.divider()


# ---------------- SIDEBAR SETTINGS ----------------
with st.sidebar:
    st.header("⚙️ Model Settings")

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower values show more detections but may include false positives.",
    )

    image_size = st.select_slider(
        "Image size",
        options=[320, 416, 512, 640, 800],
        value=640,
    )

    st.caption("Model: best.pt")
    st.caption("Supported input: JPG, JPEG, PNG")


# ---------------- VERIFY MODEL FILE ----------------
if not MODEL_PATH.is_file():
    st.error("The trained model file best.pt was not found.")
    st.write("Expected model location:")
    st.code(str(MODEL_PATH))
    st.info(
        "Place your trained best.pt file in the same folder as app.py."
    )
    st.stop()


# ---------------- LOAD MODEL ----------------
try:
    with st.spinner("Loading your trained road-damage model..."):
        model = load_model(str(MODEL_PATH))

    st.success("Custom model loaded successfully.")

    with st.expander("View model information"):
        st.write("Model file:", str(MODEL_PATH))
        st.write("Model classes:", model.names)

except Exception as error:
    st.error("Could not load best.pt.")
    st.code(str(error))
    st.stop()


st.divider()


# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with st.spinner("Detecting road damage..."):
            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )

        result = results[0]

        # YOLO's plot() returns a BGR image; Streamlit expects RGB.
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]

        with right:
            st.subheader("Detection Results")
            st.image(annotated_rgb, use_container_width=True)

        # ---------------- DETECTION SUMMARY ----------------
        st.divider()
        st.subheader("📊 Detection Summary")

        boxes = result.boxes
        detection_count = len(boxes) if boxes is not None else 0

        metric1, metric2 = st.columns(2)

        metric1.metric("Total detections", detection_count)
        metric2.metric("Confidence threshold", f"{confidence:.0%}")

        if detection_count == 0:
            st.warning(
                "No objects were detected at this confidence level. "
                "Try lowering the confidence threshold or testing another image."
            )
        else:
            detection_data = []

            for box in boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                if isinstance(model.names, dict):
                    class_name = model.names.get(
                        class_id, f"Class {class_id}"
                    )
                else:
                    class_name = model.names[class_id]

                xyxy = box.xyxy[0].tolist()

       
            detection_data.append(
                {
                    "Detected class": class_name,
                    "Confidence": f"{score:.1%}",
                    "X1": round(xyxy[0], 1),
                    "Y1": round(xyxy[1], 1),
                    "X2": round(xyxy[2], 1),
                    "Y2": round(xyxy[3], 1),
                }
            )

        st.dataframe(
            detection_data,
            use_container_width=True,
            hide_index=True,
        )

        import io

        image_buffer = io.BytesIO()
        Image.fromarray(annotated_rgb).save(
            image_buffer, format="PNG"
        )

        st.download_button(
            label="Download detected image",
            data=image_buffer.getvalue(),
            file_name="road_damage_detection.png",
            mime="image/png",
        )

    except Exception as error:
        st.error("An error occurred while processing the image.")
        st.exception(error)

else:
    st.info("Upload a road image above to start detection.")

st.divider()
st.caption(
    "RoadGuard | AI predictions are estimates. "
    "Verify detections before using them for road maintenance decisions."
)