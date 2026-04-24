# AquaVision AI — Deep-Sea Vision System

> Real-time AI-powered underwater image analysis for marine pollution detection and ecosystem health monitoring.

---

## Overview

AquaVision AI is a computer vision system that transforms raw underwater images into structured environmental intelligence. By combining YOLOv8 object detection, image enhancement pipelines, and a pollution scoring engine, the system allows researchers, NGOs, and conservation teams to assess marine ecosystem health from a single image — instantly and without manual analysis.

**Problem:** Ocean pollution is difficult to detect, expensive to monitor, and often invisible until it becomes critical. Traditional methods are slow and require significant field resources.

**Solution:** An end-to-end AI pipeline — from raw underwater image to a downloadable environmental health report — enabling faster, data-driven decisions for ocean conservation.

---

## Features

| Feature | Description |
|---|---|
| Object Detection | YOLOv8-powered detection of marine life, debris, and underwater objects |
| Image Enhancement | OpenCV detail enhancement to improve underwater image clarity |
| Pollution Scoring | Rule-based + AI hybrid engine that assigns a 0–100 pollution index |
| AI Decision System | Classifies water quality as Clean / Moderate / High Pollution |
| Caption Generation | Automatic scene description based on visual features |
| Heatmap Visualization | Highlights regions of interest using thermal mapping |
| CSV Report Export | Downloadable structured report with all metrics and timestamps |
| Dual Input Modes | Supports both file upload and live webcam capture |

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/aquavision-ai.git
cd aquavision-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will launch at `https://aquavision-3m5prm7qtkttzr22yzq2b5.streamlit.app/`.


---

## How It Works



### Pollution Scoring Logic

The pollution index (0–100) is computed using:
- **Brightness** — Low visibility is correlated with turbid, polluted water.
- **Blue Ratio** — Abnormally high blue channel dominance indicates deeper or murkier conditions.

The system is a **hybrid AI** — YOLO handles detection (learned), while the scoring layer is rule-based for interpretability and auditability.

---

## Project Structure


---

## Tech Stack

- **Frontend/UI:** Streamlit
- **Computer Vision:** OpenCV, YOLOv8 (Ultralytics)
- **Data Processing:** NumPy, Pandas
- **Image Handling:** Pillow

---

## Configuration

The app runs in two modes automatically:

- **Real Mode** — If `ultralytics` is installed and `yolov8n.pt` is available, actual YOLO inference runs.
- **Demo Mode** — If YOLO is unavailable, the app falls back to simulated detections, allowing the full pipeline to be demonstrated without GPU dependencies.

---

## Roadmap

- [ ] Train on dedicated marine datasets (TrashCan, Roboflow Underwater)
- [ ] Real-time underwater drone/video stream integration
- [ ] BLIP-based vision-language captioning
- [ ] Satellite ocean data fusion
- [ ] Advanced marine species classification
- [ ] Global marine health dashboard

---

## Use Cases

- Marine research and biodiversity studies
- Coastal pollution tracking for environmental agencies
- NGO field operations and conservation reporting
- Government environmental policy support

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is open-source. See `LICENSE` for details.
