import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import random

st.set_page_config(page_title="Deep-Sea Vision AI", layout="wide")

st.title("🌊 Deep-Sea Vision AI")
st.markdown("""
AI-powered system to:
- Enhance underwater images  
- Detect marine objects  
- Generate captions  
- Estimate pollution levels  
""")


def enhance_image(img):
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)


OBJECT_POOL = [
    "fish", "coral", "rock", "sand", "seaweed",
    "plastic bottle", "plastic bag", "net", "can"
]

def fake_detect_objects(img):
    h, w, _ = img.shape
    detections = []

    num_objects = random.randint(2, 6)

    for _ in range(num_objects):
        label = random.choice(OBJECT_POOL)
        conf = round(random.uniform(0.5, 0.95), 2)

        # Random bounding box
        x1 = random.randint(0, w//2)
        y1 = random.randint(0, h//2)
        x2 = x1 + random.randint(50, 150)
        y2 = y1 + random.randint(50, 150)

        detections.append((label, conf, (x1, y1, x2, y2)))

    return detections


def draw_boxes(img, detections):
    for label, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} ({conf})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return img


def generate_caption(detections):
    objects = [d[0] for d in detections]

    if not objects:
        return "Clear underwater scene with no visible objects."

    return f"Underwater scene showing {', '.join(objects[:3])} with varying visibility."


def pollution_index(detections):
    plastic_keywords = ["plastic", "bottle", "bag", "net", "can"]
    count = sum(1 for d in detections if any(k in d[0] for k in plastic_keywords))
    return count * 10


def agent_decision(detections):
    if not detections:
        return "No Objects Detected"

    avg_conf = sum([d[1] for d in detections]) / len(detections)

    if avg_conf < 0.6:
        return "⚠️ Recheck Image"
    elif avg_conf < 0.8:
        return "🤔 Moderate Confidence"
    else:
        return "✅ High Confidence"


uploaded_file = st.file_uploader("📤 Upload underwater image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    enhanced = enhance_image(img_cv.copy())

    detections = fake_detect_objects(enhanced)
    boxed_img = draw_boxes(enhanced.copy(), detections)

    caption = generate_caption(detections)
    pollution = pollution_index(detections)
    decision = agent_decision(detections)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.image(boxed_img, caption="AI Analysis Output", use_column_width=True)

    st.subheader("📊 Results")

    st.write("**📝 Caption:**", caption)

    st.write("**🎯 Detected Objects:**")
    for obj, conf, _ in detections:
        st.write(f"- {obj} ({conf})")

    st.write(f"**🌍 Pollution Index:** {pollution}")
    st.write(f"**🤖 AI Decision:** {decision}")

    df = pd.DataFrame(
        [(d[0], d[1]) for d in detections],
        columns=["Object", "Confidence"]
    )
    df["Pollution Index"] = pollution
    df["Timestamp"] = datetime.datetime.now()

    st.download_button(
        "📥 Download Report (CSV)",
        df.to_csv(index=False),
        file_name="deep_sea_report.csv"
    )