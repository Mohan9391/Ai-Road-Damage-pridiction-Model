import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from PIL import Image
from utils import load_model, detect_image, detect_video, get_severity_label
# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoadGuard - Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ── Session State Init ────────────────────────────────────────────────────
if "img_result" not in st.session_state:
    st.session_state.img_result = None
if "img_original" not in st.session_state:
    st.session_state.img_original = None
if "vid_result" not in st.session_state:
    st.session_state.vid_result = None
# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp {
        background: linear-gradient(145deg, #0a0a1a 0%, #111130 40%, #0d0d2b 100%);
        font-family: 'Inter', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f2e 0%, #1a1a3e 100%) !important;
        border-right: 1px solid rgba(100, 100, 255, 0.1);
    }
    .hero-container { text-align: center; padding: 2rem 1rem 1rem; }
    .hero-title {
        font-size: 3rem; font-weight: 800; margin-bottom: 0.3rem; letter-spacing: -1px;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7, #ff6ec7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-subtitle { font-size: 1.15rem; color: #8888bb; font-weight: 300; }
    .glass-card {
        background: rgba(255,255,255,0.04); backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 20px;
        padding: 1.8rem; margin-bottom: 1.2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1.2rem; margin: 1.5rem 0; }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08); border-radius: 18px;
        padding: 1.5rem; text-align: center; position: relative; overflow: hidden;
    }
    .metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #00d4ff, #7b2ff7); }
    .metric-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
    .metric-value { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #00d4ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-label { font-size: 0.85rem; color: #8888bb; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 1px; }
    .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.2rem; margin: 1.5rem 0; }
    .feature-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 1.5rem; }
    .feature-card h4 { color: #e0e0ff; font-size: 1rem; margin-bottom: 0.5rem; }
    .feature-card p { color: #7777aa; font-size: 0.88rem; line-height: 1.5; }
    .result-badge { display: inline-block; padding: 0.6rem 1.5rem; border-radius: 50px; font-weight: 600; font-size: 1.1rem; margin: 0.5rem 0; }
    .badge-safe { background: rgba(0,242,96,0.12); border: 1px solid rgba(0,242,96,0.3); color: #00f260; }
    .badge-warning { background: rgba(255,165,0,0.12); border: 1px solid rgba(255,165,0,0.3); color: #ffa500; }
    .badge-danger { background: rgba(255,65,108,0.12); border: 1px solid rgba(255,65,108,0.3); color: #ff416c; }
    .section-header { color: #d0d0ff; font-size: 1.4rem; font-weight: 600; margin: 1.5rem 0 1rem; }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent); margin: 1.5rem 0; }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7b2ff7) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 0.7rem 2rem !important; font-weight: 600 !important; font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(0,212,255,0.25) !important;
    }
    .about-block {
        background: rgba(255,255,255,0.03); border-left: 3px solid #7b2ff7;
        border-radius: 0 12px 12px 0; padding: 1.2rem 1.5rem; margin: 1rem 0;
        color: #a0a0cc; line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)
# ── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model(model_path="yolov8n.pt"):
    return load_model(model_path)
# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🛣️ RoadGuard")
    st.markdown("*AI-Powered Road Inspection*")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Home", "🔍 Detection", "📊 Analytics", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    model_choice = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], help="Nano=fastest, Medium=most accurate")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05, help="Minimum confidence for detections")
    st.markdown("---")
    st.info("💡 Upload road images or videos to detect damage using YOLOv8 deep learning.")
# Load model with spinner
with st.spinner("⏳ Loading YOLO model..."):
    model = get_model(model_choice)
if model is None:
    st.error("❌ Failed to load YOLO model. Please check your installation.")
    st.stop()
# ══════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">RoadGuard</div>
        <div class="hero-subtitle">AI-Powered Road Damage Detection System</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="text-align:center;">
        <p style="color:#a0a0cc; font-size:1.05rem; line-height:1.8; max-width:700px; margin:0 auto;">
            RoadGuard uses <strong style="color:#00d4ff;">YOLOv8</strong> deep learning
            to detect and classify road surface damage —
            <strong style="color:#ff6ec7;">cracks</strong>,
            <strong style="color:#ffa500;">potholes</strong>, and
            <strong style="color:#7b2ff7;">surface deterioration</strong> —
            from uploaded images and videos.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card"><div class="metric-icon">🤖</div><div class="metric-value">YOLOv8</div><div class="metric-label">Detection Model</div></div>
        <div class="metric-card"><div class="metric-icon">🖼️</div><div class="metric-value">Image</div><div class="metric-label">Upload Support</div></div>
        <div class="metric-card"><div class="metric-icon">🎥</div><div class="metric-value">Video</div><div class="metric-label">Upload Support</div></div>
        <div class="metric-card"><div class="metric-icon">⚡</div><div class="metric-value">Real-Time</div><div class="metric-label">Processing</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">✨ Key Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card"><h4>🔍 Smart Detection</h4><p>YOLOv8 identifies multiple damage types with bounding boxes and confidence scores.</p></div>
        <div class="feature-card"><h4>📸 Image Analysis</h4><p>Upload road images (JPG/PNG) and get instant annotated results.</p></div>
        <div class="feature-card"><h4>🎬 Video Processing</h4><p>Process dashcam videos frame-by-frame with downloadable annotated output.</p></div>
        <div class="feature-card"><h4>📊 Analytics</h4><p>View charts for class distribution, confidence scores, and severity assessment.</p></div>
        <div class="feature-card"><h4>⚙️ Configurable</h4><p>Choose model size and confidence threshold for fine-tuned sensitivity.</p></div>
        <div class="feature-card"><h4>🛡️ Severity Rating</h4><p>Automatic condition assessment from safe to severely damaged.</p></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#6666aa; padding:1.5rem;">👈 Select <strong style="color:#00d4ff;">🔍 Detection</strong> to start.</div>', unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════
#  DETECTION PAGE
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Detection":
    st.markdown("""
    <div class="hero-container" style="padding-bottom:0;">
        <div class="hero-title" style="font-size:2.2rem;">🔍 Road Damage Detection</div>
        <div class="hero-subtitle">Upload an image or video to analyze road conditions</div>
    </div>
    """, unsafe_allow_html=True)
    mode = st.radio("Mode", ["📸 Image Detection", "🎬 Video Detection"], horizontal=True, label_visibility="collapsed")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    # ── IMAGE DETECTION ────────────────────────────────────────────
    if mode == "📸 Image Detection":
        uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_up")
        if uploaded is not None:
            # KEY FIX: Use getvalue() so bytes persist across Streamlit reruns
            file_bytes = uploaded.getvalue()
            image_array = np.frombuffer(file_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image_bgr is None:
                st.error("❌ Could not read this image file. Please upload a valid image.")
            else:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                # Show uploaded image
                st.image(image_rgb, caption=f"📷 Uploaded: {uploaded.name} ({image_bgr.shape[1]}x{image_bgr.shape[0]})", use_container_width=True)
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                # Detection button
                _, btn_col, _ = st.columns([1, 2, 1])
                with btn_col:
                    run_detect = st.button("🚀 Detect Road Damage", use_container_width=True, key="run_img_detect")
                if run_detect:
                    with st.spinner("🔄 Running YOLOv8 detection..."):
                        result = detect_image(model, image_bgr, confidence)
                        st.session_state.img_result = result
                        st.session_state.img_original = image_rgb
                # ── Show Results ──
                if st.session_state.img_result is not None:
                    result = st.session_state.img_result
                    summary = result["summary"]
                    total = summary["total_detections"]
                    avg_conf = summary["avg_confidence"]
                    st.markdown('<div class="section-header">🎯 Detection Results</div>', unsafe_allow_html=True)
                    # Side-by-side comparison
                    if result["annotated_image"] is not None:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**📷 Original**")
                            st.image(st.session_state.img_original, use_container_width=True)
                        with c2:
                            st.markdown("**🔍 Detected**")
                            st.image(result["annotated_image"], use_container_width=True)
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    # Severity badge
                    severity_label, _ = get_severity_label(total)
                    badge = "badge-safe" if total == 0 else ("badge-warning" if total <= 5 else "badge-danger")
                    st.markdown(f'<div style="text-align:center;margin:1rem 0;"><div class="result-badge {badge}">{severity_label}</div></div>', unsafe_allow_html=True)
                    # Metric cards
                    st.markdown(f"""
                    <div class="metric-row">
                        <div class="metric-card"><div class="metric-icon">🎯</div><div class="metric-value">{total}</div><div class="metric-label">Total Detections</div></div>
                        <div class="metric-card"><div class="metric-icon">📊</div><div class="metric-value">{avg_conf:.1%}</div><div class="metric-label">Avg Confidence</div></div>
                        <div class="metric-card"><div class="metric-icon">🏷️</div><div class="metric-value">{len(summary['class_counts'])}</div><div class="metric-label">Classes Found</div></div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Detection table
                    if result["detections"]:
                        st.markdown('<div class="section-header">📋 Detection Details</div>', unsafe_allow_html=True)
                        # Class summary
                        sum_df = pd.DataFrame(list(summary["class_counts"].items()), columns=["Class", "Count"]).sort_values("Count", ascending=False)
                        st.dataframe(sum_df, use_container_width=True, hide_index=True)
                        # Full list
                        with st.expander("🔎 View All Detections"):
                            rows = []
                            for i, d in enumerate(result["detections"], 1):
                                rows.append({"#": i, "Class": d["class_name"], "Confidence": f"{d['confidence']:.2%}",
                                             "BBox": f"({d['bbox'][0]:.0f}, {d['bbox'][1]:.0f}, {d['bbox'][2]:.0f}, {d['bbox'][3]:.0f})"})
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    elif total == 0:
                        st.success("✅ No objects detected. Try lowering the confidence threshold in the sidebar.")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <div style="font-size:4rem; margin-bottom:1rem;">📸</div>
                <p style="color:#8888bb; font-size:1.1rem;">
                    Upload a road surface image to begin detection.<br><br>
                    Supported: <strong style="color:#00d4ff;">JPG, JPEG, PNG, BMP, WEBP</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.img_result = None
    # ── VIDEO DETECTION ────────────────────────────────────────────
    elif mode == "🎬 Video Detection":
        uploaded_vid = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
        if uploaded_vid is not None:
            # Save video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.getvalue())
            tfile.close()
            video_path = tfile.name
            st.video(video_path)
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                run_vid = st.button("🚀 Detect Damage in Video", use_container_width=True, key="run_vid_detect")
            if run_vid:
                progress = st.progress(0, text="🔄 Analyzing video frames...")
                def update(pct):
                    progress.progress(min(pct, 1.0), text=f"Analyzing... {pct:.0%}")
                try:
                    vid_result = detect_video(model, video_path, confidence, progress_callback=update)
                    progress.progress(1.0, text="✅ Complete!")
                    st.session_state.vid_result = vid_result
                except Exception as e:
                    st.error(f"❌ Video processing error: {e}")
                finally:
                    try:
                        os.unlink(video_path)
                    except Exception:
                        pass
            # Show video results
            if st.session_state.vid_result is not None:
                vr = st.session_state.vid_result
                vs = vr["overall_summary"]
                total = vs["total_detections"]
                avg_per_frame = total / max(vs["total_frames"], 1)
                severity_label, _ = get_severity_label(int(avg_per_frame))
                badge = "badge-safe" if avg_per_frame == 0 else ("badge-warning" if avg_per_frame <= 3 else "badge-danger")
                st.markdown(f'<div style="text-align:center;margin:1rem 0;"><div class="result-badge {badge}">{severity_label}</div></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card"><div class="metric-icon">🎞️</div><div class="metric-value">{vs['total_frames']}</div><div class="metric-label">Frames</div></div>
                    <div class="metric-card"><div class="metric-icon">🎯</div><div class="metric-value">{total}</div><div class="metric-label">Detections</div></div>
                    <div class="metric-card"><div class="metric-icon">📊</div><div class="metric-value">{vs['avg_confidence']:.1%}</div><div class="metric-label">Avg Confidence</div></div>
                    <div class="metric-card"><div class="metric-icon">⚡</div><div class="metric-value">{vs['fps']}</div><div class="metric-label">FPS</div></div>
                </div>
                """, unsafe_allow_html=True)
                if vs["class_counts"]:
                    st.markdown('<div class="section-header">📋 Class Summary</div>', unsafe_allow_html=True)
                    vid_df = pd.DataFrame(list(vs["class_counts"].items()), columns=["Class", "Count"]).sort_values("Count", ascending=False)
                    st.dataframe(vid_df, use_container_width=True, hide_index=True)
                output_path = vr["output_path"]
                if os.path.exists(output_path):
                    st.markdown('<div class="section-header">🎬 Annotated Video</div>', unsafe_allow_html=True)
                    st.video(output_path)
                    with open(output_path, "rb") as f:
                        st.download_button("⬇️ Download Annotated Video", f, "roadguard_output.mp4", "video/mp4", use_container_width=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <div style="font-size:4rem; margin-bottom:1rem;">🎬</div>
                <p style="color:#8888bb; font-size:1.1rem;">
                    Upload a road video to begin detection.<br><br>
                    Supported: <strong style="color:#00d4ff;">MP4, AVI, MOV, MKV</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════
#  ANALYTICS PAGE
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("""
    <div class="hero-container" style="padding-bottom:0;">
        <div class="hero-title" style="font-size:2.2rem;">📊 Analytics Dashboard</div>
        <div class="hero-subtitle">Visual analysis of detection results</div>
    </div>
    """, unsafe_allow_html=True)
    has_img = st.session_state.img_result is not None
    has_vid = st.session_state.vid_result is not None
    if not has_img and not has_vid:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div style="font-size:4rem; margin-bottom:1rem;">📊</div>
            <p style="color:#8888bb; font-size:1.1rem;">
                No results yet. Go to <strong style="color:#00d4ff;">🔍 Detection</strong> first.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        if has_img:
            result = st.session_state.img_result
            summary = result["summary"]
            st.markdown('<div class="section-header">📸 Image Analytics</div>', unsafe_allow_html=True)
            if summary["total_detections"] > 0:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Class Distribution**")
                    classes = list(summary["class_counts"].keys())
                    counts = list(summary["class_counts"].values())
                    fig, ax = plt.subplots(figsize=(7, 4))
                    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
                    bars = ax.barh(classes, counts, color=colors, edgecolor="white", linewidth=0.5)
                    for bar, c in zip(bars, counts):
                        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(c), va="center", color="#e0e0ff", fontweight="bold")
                    ax.set_xlabel("Count", color="#a0a0dd"); ax.set_title("Detections by Class", color="#e0e0ff", pad=10)
                    ax.tick_params(colors="#a0a0dd"); fig.patch.set_facecolor("#0a0a1a"); ax.set_facecolor("#0a0a1a")
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                with c2:
                    st.markdown("**Confidence Distribution**")
                    confs = [d["confidence"] for d in result["detections"]]
                    fig2, ax2 = plt.subplots(figsize=(7, 4))
                    ax2.hist(confs, bins=15, color="#00d4ff", alpha=0.7, edgecolor="white")
                    ax2.axvline(np.mean(confs), color="#ff6ec7", linestyle="--", label=f"Mean: {np.mean(confs):.2f}")
                    ax2.set_xlabel("Confidence", color="#a0a0dd"); ax2.set_ylabel("Count", color="#a0a0dd")
                    ax2.set_title("Confidence Scores", color="#e0e0ff", pad=10); ax2.legend(facecolor="#0a0a1a")
                    ax2.tick_params(colors="#a0a0dd"); fig2.patch.set_facecolor("#0a0a1a"); ax2.set_facecolor("#0a0a1a")
                    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
                if len(classes) > 1:
                    st.markdown("**Class Proportions**")
                    fig3, ax3 = plt.subplots(figsize=(5, 5))
                    ax3.pie(counts, labels=classes, autopct="%1.1f%%", colors=plt.cm.Set2(np.linspace(0, 1, len(classes))),
                            textprops={"color": "#e0e0ff"}, wedgeprops={"edgecolor": "#0a0a1a", "linewidth": 2})
                    fig3.patch.set_facecolor("#0a0a1a"); plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)
            else:
                st.success("✅ No damage in last image analysis.")
        if has_vid:
            vr = st.session_state.vid_result
            vs = vr["overall_summary"]
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🎬 Video Analytics</div>', unsafe_allow_html=True)
            if vs["total_detections"] > 0:
                all_dets = vr["all_detections"]
                frames = [d["frame"] for d in all_dets]
                frame_counts = {}
                for f in frames:
                    frame_counts[f] = frame_counts.get(f, 0) + 1
                fns = sorted(frame_counts.keys())
                dcs = [frame_counts[f] for f in fns]
                st.markdown("**Detections Over Time**")
                fig4, ax4 = plt.subplots(figsize=(12, 3.5))
                ax4.fill_between(fns, dcs, alpha=0.3, color="#00d4ff")
                ax4.plot(fns, dcs, color="#00d4ff", linewidth=1.5)
                ax4.set_xlabel("Frame", color="#a0a0dd"); ax4.set_ylabel("Detections", color="#a0a0dd")
                ax4.set_title("Detection Timeline", color="#e0e0ff", pad=10)
                ax4.tick_params(colors="#a0a0dd"); fig4.patch.set_facecolor("#0a0a1a"); ax4.set_facecolor("#0a0a1a")
                plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)
                c1, c2 = st.columns(2)
                with c1:
                    v_classes = list(vs["class_counts"].keys()); v_counts = list(vs["class_counts"].values())
                    fig5, ax5 = plt.subplots(figsize=(7, 4))
                    ax5.barh(v_classes, v_counts, color=plt.cm.magma(np.linspace(0.3, 0.8, len(v_classes))), edgecolor="white")
                    ax5.set_xlabel("Count", color="#a0a0dd"); ax5.set_title("Video Classes", color="#e0e0ff", pad=10)
                    ax5.tick_params(colors="#a0a0dd"); fig5.patch.set_facecolor("#0a0a1a"); ax5.set_facecolor("#0a0a1a")
                    plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)
                with c2:
                    v_confs = [d["confidence"] for d in all_dets]
                    fig6, ax6 = plt.subplots(figsize=(7, 4))
                    ax6.hist(v_confs, bins=20, color="#7b2ff7", alpha=0.7, edgecolor="white")
                    ax6.axvline(np.mean(v_confs), color="#ff6ec7", linestyle="--", label=f"Mean: {np.mean(v_confs):.2f}")
                    ax6.set_xlabel("Confidence", color="#a0a0dd"); ax6.set_title("Confidence (Video)", color="#e0e0ff", pad=10)
                    ax6.legend(facecolor="#0a0a1a"); ax6.tick_params(colors="#a0a0dd")
                    fig6.patch.set_facecolor("#0a0a1a"); ax6.set_facecolor("#0a0a1a")
                    plt.tight_layout(); st.pyplot(fig6); plt.close(fig6)
            else:
                st.success("✅ No damage in last video analysis.")
# ══════════════════════════════════════════════════════════════════════════
#  ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div class="hero-container" style="padding-bottom:0;">
        <div class="hero-title" style="font-size:2.2rem;">ℹ️ About RoadGuard</div>
        <div class="hero-subtitle">System overview and documentation</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">🛣️ Project Overview</div>
        <div class="about-block">
            <strong>RoadGuard</strong> is an AI-powered road damage detection system using
            YOLOv8 deep learning to identify and classify road surface damage from images and videos.
            It helps municipal authorities and road maintenance crews prioritize repairs.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">🤖 Model Details</div>
        <div class="about-block">
            <strong>YOLOv8</strong> by <a href="https://ultralytics.com" style="color:#00d4ff;">Ultralytics</a><br><br>
            <ul>
                <li><strong>Architecture:</strong> CSPDarknet + PANet + Detection Head</li>
                <li><strong>Sizes:</strong> Nano, Small, Medium</li>
                <li><strong>Input:</strong> Images (JPG/PNG) & Videos (MP4/AVI/MOV)</li>
                <li><strong>Output:</strong> Bounding boxes, class labels, confidence scores</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">🏗️ Road Damage Classes (RDD2022)</div>
        <div class="about-block">
            When fine-tuned on road damage datasets:
            <ul>
                <li>🔶 <strong>D00</strong> — Longitudinal Crack</li>
                <li>🔴 <strong>D10</strong> — Transverse Crack</li>
                <li>🟡 <strong>D20</strong> — Alligator Crack</li>
                <li>🟣 <strong>D40</strong> — Pothole</li>
            </ul>
            <br><em>Default YOLOv8 detects general objects (cars, people, etc.). For road-specific detection, use a fine-tuned model.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">⚠️ Disclaimer</div>
        <div class="about-block" style="border-left-color:#ff416c;">
            <strong>Educational Purpose Only.</strong> Do not use as the sole basis for road maintenance decisions.
            Always complement AI analysis with professional engineering inspections.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="margin-top:0;">🙏 Built With</div>
        <div class="about-block">
            <a href="https://streamlit.io/" style="color:#00d4ff;">Streamlit</a> •
            <a href="https://ultralytics.com/" style="color:#00d4ff;">YOLOv8</a> •
            <a href="https://opencv.org/" style="color:#00d4ff;">OpenCV</a> •
            <a href="https://matplotlib.org/" style="color:#00d4ff;">Matplotlib</a> •
            <a href="https://pandas.pydata.org/" style="color:#00d4ff;">Pandas</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
