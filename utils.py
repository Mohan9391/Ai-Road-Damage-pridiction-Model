import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
def load_model(model_path="yolov8n.pt"):
    """Load a YOLO model. Downloads automatically if not found locally."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
def detect_image(model, image_bgr, confidence=0.25):
    """
    Run YOLO detection on a single BGR image.
    Returns dict with annotated_image (RGB), detections list, and summary.
    """
    if model is None:
        return {"annotated_image": None, "detections": [], "summary": {"total_detections": 0, "class_counts": {}, "avg_confidence": 0}}
    try:
        # Run YOLO inference
        results = model.predict(source=image_bgr, conf=confidence, verbose=False)
        result = results[0]
        # Get annotated image (YOLO returns BGR, convert to RGB)
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        # Parse all detections
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                cls_name = result.names.get(cls_id, f"Class_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf_val,
                    "bbox": xyxy,
                })
        # Build summary
        class_counts = {}
        for d in detections:
            name = d["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1
        avg_conf = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0
        return {
            "annotated_image": annotated_rgb,
            "detections": detections,
            "summary": {
                "total_detections": len(detections),
                "class_counts": class_counts,
                "avg_confidence": avg_conf,
            },
        }
    except Exception as e:
        print(f"Detection error: {e}")
        return {"annotated_image": None, "detections": [], "summary": {"total_detections": 0, "class_counts": {}, "avg_confidence": 0}}
def detect_video(model, video_path, confidence=0.25, progress_callback=None):
    """
    Run YOLO detection on every frame of a video.
    Returns dict with output_path, all_detections, and overall_summary.
    """
    if model is None:
        return {"output_path": "", "all_detections": [], "overall_summary": {"total_frames": 0, "total_detections": 0, "class_counts": {}, "avg_confidence": 0, "fps": 0}}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Output video file
    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    all_detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=confidence, verbose=False)
        result = results[0]
        # Collect detections
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                cls_name = result.names.get(cls_id, f"Class_{cls_id}")
                all_detections.append({
                    "frame": frame_idx,
                    "class_name": cls_name,
                    "confidence": conf_val,
                })
        # Write annotated frame
        annotated = result.plot()
        writer.write(annotated)
        frame_idx += 1
        if progress_callback and total_frames > 0:
            progress_callback(frame_idx / total_frames)
    cap.release()
    writer.release()
    # Summary
    class_counts = {}
    for d in all_detections:
        name = d["class_name"]
        class_counts[name] = class_counts.get(name, 0) + 1
    avg_conf = float(np.mean([d["confidence"] for d in all_detections])) if all_detections else 0.0
    return {
        "output_path": output_path,
        "all_detections": all_detections,
        "overall_summary": {
            "total_frames": frame_idx,
            "total_detections": len(all_detections),
            "class_counts": class_counts,
            "avg_confidence": avg_conf,
            "fps": fps,
        },
    }
def get_severity_label(count):
    """Return severity label and color based on detection count."""
    if count == 0:
        return "✅ No Damage Detected", "green"
    elif count <= 3:
        return "⚠️ Minor Damage", "orange"
    elif count <= 8:
        return "🔶 Moderate Damage", "darkorange"
    else:
        return "🚨 Severe Damage", "red"

