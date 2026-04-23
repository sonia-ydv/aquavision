import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import numpy as np
import datetime

st.set_page_config(page_title="Deep-Sea Vision AI", layout="wide")

st.title("🌊 Deep-Sea Vision AI")
st.markdown("""
AI-powered tool to:
- Enhance underwater images  
- Detect marine objects  
- Estimate pollution levels  
""")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  

yolo_model = load_yolo()

ENABLE_CAPTION = False  

if ENABLE_CAPTION:
    from transformers import BlipProcessor, BlipForConditionalGeneration

    @st.cache_resource
    def load_caption_model():
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model

    processor, caption_model = load_caption_model()

def enhance_image(img):
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

def detect_objects(img):
    results = yolo_model(img)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo_model.names[cls]
            detections.append((label, conf))
    return detections, results

def pollution_index(detections):
    plastic_keywords = ["bottle", "plastic", "bag"]
    count = sum(1 for d in detections if any(k in d[0].lower() for k in plastic_keywords))
    return count * 10

def agent_decision(detections):
    if not detections:
        return "No Objects Detected"
    avg_conf = sum([d[1] for d in detections]) / len(detections)
    if avg_conf < 0.4:
        return "⚠️ Recheck Image"
    elif avg_conf < 0.7:
        return "🤔 Low Confidence"
    else:
        return "✅ High Confidence"


uploaded_file = st.file_uploader("📤 Upload underwater image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    enhanced = enhance_image(img_cv)

   
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    with col2:
        st.image(enhanced_rgb, caption="Enhanced Image", use_column_width=True)

    
    with st.spinner("🔍 Analyzing image..."):
        try:
            detections, results = detect_objects(enhanced_rgb)
            pollution = pollution_index(detections)
            decision = agent_decision(detections)

            if ENABLE_CAPTION:
                inputs = processor(image, return_tensors="pt")
                out = caption_model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
            else:
                caption = "Caption disabled for performance"

        except Exception as e:
            st.error("Processing failed. Try another image.")
            st.stop()

    st.success("Analysis Complete ✅")

    
    st.subheader("📊 Results")

    st.write("**📝 Caption:**", caption)

    
    annotated = results[0].plot()
    st.image(annotated, caption="Detection Output")

    # Table format
    if detections:
        df = pd.DataFrame(detections, columns=["Object", "Confidence"])
        st.dataframe(df)
    else:
        st.write("No objects detected")

    st.write(f"**🌍 Pollution Index:** {pollution}")
    st.write(f"**🤖 AI Decision:** {decision}")

    df_export = pd.DataFrame(detections, columns=["Object", "Confidence"])
    df_export["Pollution Index"] = pollution
    df_export["Timestamp"] = datetime.datetime.now()

    st.download_button(
        "📥 Download Report (CSV)",
        df_export.to_csv(index=False),
        file_name="deep_sea_report.csv"
    )