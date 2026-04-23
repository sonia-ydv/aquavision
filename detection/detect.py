from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pretrained

def detect_objects(image):
    results = model(image)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append(label)

    return detections