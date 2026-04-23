import cv2
import pandas as pd
from enhancement.enhance import enhance_image
from detection.detect import detect_objects
from utils.scoring import calculate_score

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    data = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        enhanced = frame  # (skip heavy enhancement for speed)

        detections = detect_objects(enhanced)

        counts = {}
        for obj in detections:
            counts[obj] = counts.get(obj, 0) + 1

        score = calculate_score(counts)

        data.append({
            "frame": frame_id,
            "fish": counts.get("fish", 0),
            "plastic": counts.get("plastic", 0),
            "score": score
        })

        frame_id += 1

    df = pd.DataFrame(data)
    df.to_csv("output/report.csv", index=False)

    return df