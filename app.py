import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import io

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "best.pt")
CLASS_NAMES = {
    0: "Corrosion",
    1: "Wreckage",
    2: "Marine Growth",
    3: "Trash",
    4: "Pipeline"
}
CLASS_COLORS = {
    0: (255, 0, 0),      # Red
    1: (0, 255, 0),      # Green
    2: (0, 165, 255),    # Orange
    3: (255, 255, 0),    # Yellow
    4: (255, 0, 255)     # Magenta
}

# ── Page Setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Underwater Anomaly Detection",
    page_icon  = "🌊",
    layout     = "wide"
)

st.title("🌊 Underwater Anomaly Detection System")
st.markdown("**AI-powered subsea inspection for ROV/AUV operators**")
st.divider()

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ── Turbidity Filter ───────────────────────────────────────────────────────────
def apply_turbidity(image, level):
    """Simulate underwater turbidity/green tint"""
    if level == "None":
        return image
    overlay = image.copy()
    if level == "Low":
        green_tint = np.full_like(image, (0, 30, 0))
        blur = 1
        alpha = 0.1
    elif level == "Medium":
        green_tint = np.full_like(image, (0, 60, 10))
        blur = 3
        alpha = 0.25
    else:  # High
        green_tint = np.full_like(image, (0, 90, 20))
        blur = 7
        alpha = 0.4
    overlay = cv2.addWeighted(image, 1 - alpha, green_tint, alpha, 0)
    if blur > 1:
        overlay = cv2.GaussianBlur(overlay, (blur, blur), 0)
    return overlay

# ── Run Detection ──────────────────────────────────────────────────────────────
def run_detection(image_bgr, conf_threshold):
    results = model.predict(
        source = image_bgr,
        conf   = conf_threshold,
        verbose = False
    )
    return results[0]

# ── Draw Boxes ─────────────────────────────────────────────────────────────────
def draw_boxes(image_bgr, result):
    annotated = image_bgr.copy()
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = CLASS_COLORS.get(cls, (255, 255, 255))
            label = f"{CLASS_NAMES.get(cls, 'Unknown')} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )
            detections.append({
                "class"     : CLASS_NAMES.get(cls, "Unknown"),
                "confidence": round(conf, 3),
                "bbox"      : [x1, y1, x2, y2]
            })

    return annotated, detections

# ── Generate PDF ───────────────────────────────────────────────────────────────
def generate_pdf(original_img, annotated_img, detections, filename):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    title_style = ParagraphStyle(
        "title",
        parent    = styles["Title"],
        fontSize  = 20,
        textColor = colors.HexColor("#0077B6"),
        spaceAfter = 12
    )
    story.append(Paragraph("Underwater Anomaly Detection Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"File: {filename}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # Summary
    story.append(Paragraph("Detection Summary", styles["Heading2"]))
    story.append(Paragraph(f"Total anomalies detected: <b>{len(detections)}</b>", styles["Normal"]))

    # Count per class
    class_counts = {}
    for d in detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1
    for cls, count in class_counts.items():
        story.append(Paragraph(f"• {cls}: {count}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    # Annotated image
    story.append(Paragraph("Annotated Detection Image", styles["Heading2"]))
    ann_pil  = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    img_buf  = io.BytesIO()
    ann_pil.save(img_buf, format="JPEG")
    img_buf.seek(0)
    story.append(RLImage(img_buf, width=5 * inch, height=3.5 * inch))
    story.append(Spacer(1, 0.3 * inch))

    # Detection table
    if detections:
        story.append(Paragraph("Detailed Detection Results", styles["Heading2"]))
        table_data = [["#", "Class", "Confidence", "Bounding Box"]]
        for i, d in enumerate(detections, 1):
            bbox_str = f"({d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]})"
            table_data.append([
                str(i),
                d["class"],
                f"{d['confidence']:.1%}",
                bbox_str
            ])

        table = Table(table_data, colWidths=[0.4*inch, 1.5*inch, 1.2*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0077B6")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF4FB")]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(table)

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "This report was generated automatically by the Underwater Anomaly Detection System powered by YOLOv12.",
        styles["Italic"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.9, 0.25, 0.05
)
turbidity = st.sidebar.selectbox(
    "Turbidity Simulation",
    ["None", "Low", "Medium", "High"]
)
st.sidebar.divider()
st.sidebar.markdown("**Classes:**")
for cls, name in CLASS_NAMES.items():
    color = CLASS_COLORS[cls]
    st.sidebar.markdown(
        f'<span style="color:rgb{color}">■</span> {name}',
        unsafe_allow_html=True
    )

# ── Main Upload Area ───────────────────────────────────────────────────────────
upload_type = st.radio(
    "Select input type:",
    ["Image", "Video"],
    horizontal=True
)

if upload_type == "Image":
    uploaded = st.file_uploader(
        "Upload an underwater image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        # Read image
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Apply turbidity
        image_bgr = apply_turbidity(image_bgr, turbidity)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

        # Run detection
        with st.spinner("Running detection..."):
            result     = run_detection(image_bgr, conf_threshold)
            annotated, detections = draw_boxes(image_bgr, result)

        with col2:
            st.subheader(f"Detection Results ({len(detections)} found)")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

        # Detection details
        if detections:
            st.divider()
            st.subheader("📋 Detected Anomalies")
            cols = st.columns(len(detections) if len(detections) <= 4 else 4)
            for i, det in enumerate(detections):
                with cols[i % 4]:
                    st.metric(
                        label = det["class"],
                        value = f"{det['confidence']:.1%}"
                    )
        else:
            st.info("No anomalies detected. Try lowering the confidence threshold.")

        # PDF Download
        st.divider()
        st.subheader("📄 Download Report")
        pdf_buffer = generate_pdf(
            image_bgr, annotated,
            detections, uploaded.name
        )
        st.download_button(
            label     = "⬇️ Download PDF Report",
            data      = pdf_buffer,
            file_name = f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime      = "application/pdf"
        )

elif upload_type == "Video":
    uploaded = st.file_uploader(
        "Upload a short underwater video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        cap        = cv2.VideoCapture(tfile.name)
        fps        = cap.get(cv2.CAP_PROP_FPS)
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stframe    = st.empty()
        progress   = st.progress(0)
        all_detections = []

        st.info(f"Processing {total} frames at {fps:.1f} FPS...")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = apply_turbidity(frame, turbidity)
            result = run_detection(frame, conf_threshold)
            annotated, detections = draw_boxes(frame, result)
            all_detections.extend(detections)

            stframe.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )
            frame_count += 1
            progress.progress(min(frame_count / total, 1.0))

        cap.release()
        os.unlink(tfile.name)

        st.success(f"Video processed. Total detections: {len(all_detections)}")

        # PDF for video
        st.divider()
        if all_detections:
            pdf_buffer = generate_pdf(
                annotated, annotated,
                all_detections, uploaded.name
            )
            st.download_button(
                label     = "⬇️ Download PDF Report",
                data      = pdf_buffer,
                file_name = f"video_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime      = "application/pdf"
            )