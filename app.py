import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import datetime

st.set_page_config(page_title="Deep-Sea Vision AI", layout="wide")

st.title(" AquaVision AI")


@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model, True
    except Exception as e:
        return None, False

yolo_model, YOLO_AVAILABLE = load_yolo()


def enhance_image(img):
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

def analyze_image(img):
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    b, g, r = cv2.split(img)
    blue_ratio = np.mean(b) / (np.mean(r) + 1e-5)

    return h, w, brightness, blue_ratio

def pollution_score(brightness, blue_ratio):
    score = 0
    if brightness < 60:
        score += 30
    if blue_ratio > 1.3:
        score += 20
    return min(score, 100)

def ai_decision(score):
    if score < 30:
        return "✅ Clean Water"
    elif score < 60:
        return "⚠️ Moderate Pollution"
    else:
        return "🚨 High Pollution"

def generate_caption(brightness, blue_ratio):
    if blue_ratio > 1.5:
        return "Deep underwater scene"
    elif brightness < 50:
        return "Low visibility underwater"
    else:
        return "Clear underwater scene"


def fake_detection(img):
    h, w, _ = img.shape
    boxes = []

    for _ in range(np.random.randint(1, 3)):
        x1 = np.random.randint(0, w//2)
        y1 = np.random.randint(0, h//2)
        x2 = x1 + np.random.randint(50, 120)
        y2 = y1 + np.random.randint(50, 120)

        label = np.random.choice(["fish", "plastic", "rock"])
        boxes.append(label)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return img, boxes


def real_detection(img):
    results = yolo_model(img)
    labels = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            labels.append(label)

        img = r.plot()

    return img, labels


def heatmap(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heat, 0.4, 0)


input_mode = st.radio("📥 Input", ["Upload", "Webcam"])

image = None

if input_mode == "Upload":
    file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    if file:
        image = Image.open(file)

else:
    cam = st.camera_input("Take a picture")
    if cam:
        image = Image.open(cam)


if image:
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_column_width=True)

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    enhanced = enhance_image(img_cv)

    if YOLO_AVAILABLE:
        st.success("🟢 Real YOLO Model Active")
        enhanced, labels = real_detection(enhanced)
    else:
        st.warning("🟡 Demo Mode (YOLO not available)")
        enhanced, labels = fake_detection(enhanced)

    with col2:
        st.image(enhanced, caption="Processed", use_column_width=True)

    h, w, brightness, blue_ratio = analyze_image(enhanced)
    score = pollution_score(brightness, blue_ratio)

    st.subheader("📊 Results")
    st.write("**Caption:**", generate_caption(brightness, blue_ratio))
    st.write("**Resolution:**", f"{w} x {h}")
    st.write("**Brightness:**", round(brightness,2))
    st.write("**Blue Ratio:**", round(blue_ratio,2))
    st.write("**Pollution Score:**", score)
    st.write("**Decision:**", ai_decision(score))

    st.write("**Detected Objects:**")
    if labels:
        for l in labels:
            st.write("-", l)
    else:
        st.write("No objects detected")

    st.subheader("🔥 Heatmap")
    st.image(heatmap(enhanced), use_column_width=True)

    # CSV
    df = pd.DataFrame([{
        "Width": w,
        "Height": h,
        "Brightness": brightness,
        "BlueRatio": blue_ratio,
        "Score": score,
        "Objects": ", ".join(labels),
        "Time": datetime.datetime.now()
    }])

    st.download_button("📥 Download Report", df.to_csv(index=False), "report.csv")