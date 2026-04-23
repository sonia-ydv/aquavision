import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import datetime


st.set_page_config(page_title="Deep-Sea Vision AI", layout="wide")

st.title("🌊 Deep-Sea Vision AI")
st.markdown("""
AI-powered tool to:
- Detect marine objects  
- Estimate pollution levels  
- Generate simple insights  
""")


@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # auto-downloads
    return model

model = load_model()


def detect_objects(image):
    results = model(image)
    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            detections.append((label, conf))
    
    return results, detections


def pollution_index(detections):
    keywords = ["bottle", "plastic", "bag"]
    count = sum(1 for d in detections if any(k in d[0].lower() for k in keywords))
    return count * 10


def agent_decision(detections):
    if not detections:
        return "No Objects Detected"
    
    avg_conf = sum(d[1] for d in detections) / len(detections)
    
    if avg_conf < 0.4:
        return "⚠️ Recheck Image"
    elif avg_conf < 0.7:
        return "🤔 Low Confidence"
    else:
        return "✅ High Confidence"



uploaded_file = st.file_uploader("📤 Upload underwater image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("🔍 Running AI detection..."):
        try:
            results, detections = detect_objects(img_array)
            pollution = pollution_index(detections)
            decision = agent_decision(detections)

            # Get plotted image (with bounding boxes)
            plotted_img = results[0].plot()

        except Exception as e:
            st.error("Error processing image. Try another image.")
            st.stop()

    with col2:
        st.image(plotted_img, caption="Detected Objects", use_column_width=True)

    
    st.subheader("📊 Results")

    st.write("**🎯 Detected Objects:**")
    if detections:
        for obj, conf in detections:
            st.write(f"- {obj} ({conf:.2f})")
    else:
        st.write("No objects detected")

    st.write(f"**🌍 Pollution Index:** {pollution}")
    st.write(f"**🤖 AI Decision:** {decision}")

    
    df = pd.DataFrame(detections, columns=["Object", "Confidence"])
    df["Pollution Index"] = pollution
    df["Timestamp"] = datetime.datetime.now()

    st.download_button(
        "📥 Download Report (CSV)",
        df.to_csv(index=False),
        file_name="deep_sea_report.csv"
    )