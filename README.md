# 🌊 Underwater Anomaly Detection System

AI-powered subsea inspection tool for ROV/AUV operators built with YOLOv12 and Streamlit.

**🔗 Live Demo → [underwater-anomaly-detection.streamlit.app](https://underwater-anomaly-detection.streamlit.app)**

## Features
- Upload image or video for detection
- Detects 5 classes: Corrosion, Wreckage, Marine Growth, Trash, Pipeline
- Turbidity simulation filter
- Automatic PDF report generation
- Adjustable confidence threshold

## Classes
| ID | Class | Dataset |
|---|---|---|
| 0 | Corrosion | MaVeCoDD |
| 1 | Wreckage | SUIM |
| 2 | Marine Growth | SUIM |
| 3 | Trash | TrashCan |
| 4 | Pipeline | MarinaPipe |

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model
- Architecture: YOLOv12n
- Training images: 1,000
- mAP50: 0.403 (overall)
- Best class: Pipeline (0.882 mAP50)

## Datasets Used
- MaVeCoDD (Mendeley)
- SUIM (GitHub)
- TrashCan Dataset
- MarinaPipe Dataset
