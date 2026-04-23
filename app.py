import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import numpy as np
import datetime

# Load models
yolo_model = YOLO("yolov8n.pt")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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
    return detections

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def pollution_index(detections):
    plastic_count = sum(1 for d in detections if "bottle" in d[0] or "plastic" in d[0])
    return plastic_count * 10

def agent_decision(detections):
    if not detections:
        return "No Objects Detected"
    avg_conf = sum([d[1] for d in detections]) / len(detections)
    if avg_conf < 0.4:
        return "Recheck Image ⚠️"
    elif avg_conf < 0.7:
        return "Accept with Low Confidence 🤔"
    else:
        return "High Confidence ✅"

st.title("🌊 Deep-Sea Vision Prototype")

uploaded_file = st.file_uploader("Upload underwater image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image")

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    enhanced = enhance_image(img_cv)
    st.image(enhanced, caption="Enhanced Image")

    detections = detect_objects(enhanced)
    caption = generate_caption(image)
    pollution = pollution_index(detections)
    decision = agent_decision(detections)

    st.write("### Results")
    st.write("Caption:", caption)
    st.write("Detections:", detections)
    st.write("Pollution Index:", pollution)
    st.write("Agent Decision:", decision)

    df = pd.DataFrame(detections, columns=["Object", "Confidence"])
    df["Pollution Index"] = pollution
    df["Timestamp"] = datetime.datetime.now()

    st.download_button("Download CSV", df.to_csv(index=False), "report.csv")
