
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🤖",
    layout="wide"
)

# Dashboard styling
st.markdown("""
<style>
.main {
    background-color: #101827;
}
h1, h2, h3 {
    color: #60A5FA;
}
.stMetric {
    background-color: #1E293B;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# Sidebar
st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider(
    "Detection Confidence",
    0.10, 0.90, 0.40, 0.05
)

st.sidebar.info("AI Model: YOLOv8n")

# Dashboard heading
st.title("🤖 AI OBJECT DETECTION SYSTEM")
st.write("Detect, identify and count objects using Artificial Intelligence.")

st.divider()

# Image upload
st.subheader("📤 Upload Your Image")

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Live Camera"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

with tab2:
    camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    uploaded_file = camera_image

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    if st.button("🔍 Detect Objects", type="primary"):

        with st.spinner("AI is detecting objects..."):

            image_array = np.array(image)

            results = model(
                image_array,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            # Draw bounding boxes
            output = result.plot()
            output = cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB
            )

            # Count objects
            counts = Counter()
            detections = []

            for box in result.boxes:
                class_id = int(box.cls[0])
                name = model.names[class_id]
                score = float(box.conf[0]) * 100

                counts[name] += 1

                detections.append({
                    "Object": name,
                    "Confidence": f"{score:.2f}%"
                })

        st.success("Detection completed!")

        # Statistics
        total = len(detections)
        unique = len(counts)

        average_confidence = (
            sum(float(d["Confidence"].replace("%", ""))
                for d in detections) / total
            if total else 0
        )

        st.subheader("📊 Detection Statistics")

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Objects", total)
        c2.metric("Unique Classes", unique)
        c3.metric("Average Confidence", f"{average_confidence:.2f}%")

        st.divider()

        # Images
        st.subheader("🖼️ Detection Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                output,
                caption="AI Detection Result",
                use_container_width=True
            )

        st.divider()

        # Object counts
        st.subheader("📦 Detected Objects")

        if counts:
            cols = st.columns(min(len(counts), 3))

            for i, (name, count) in enumerate(counts.items()):
                with cols[i % len(cols)]:
                    st.metric(name.title(), count)

            st.bar_chart(dict(counts))

            st.subheader("🎯 Confidence Scores")
            st.dataframe(
                detections,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning("No objects detected. Try another image.")

else:
    st.info("Upload an image to start object detection.")
    st.divider()

with st.expander("ℹ️ About This Project"):
    st.subheader("AI Object Detection System")

    st.write("""
    This project uses YOLOv8, a deep learning model,
    to detect and identify objects in images.

    Features:
    - Real-time object detection
    - Multiple object identification
    - Object counting
    - Confidence score calculation
    - Image upload and camera support
    """)

    st.write("Technology Used:")
    st.write("Python | Streamlit | OpenCV | YOLOv8")
