# 🎓 Classroom Engagement Analyzer

Real-time classroom engagement analysis using **Computer Vision**, combining **MediaPipe FaceMesh**, **YOLOv8 Pose Estimation**, and **Object Detection** to infer student attention, distraction, and fatigue with confidence scoring.

> 🚀 Built as part of Smart India Hackathon (SIH) project evolution and productionized for deployment and portfolio use.

---

## ✨ Key Features

✅ Real-time face landmark detection (MediaPipe FaceMesh)
✅ Body posture analysis using YOLOv8 Pose
✅ Phone distraction detection using YOLOv8 Object Detection
✅ Multi-signal engagement classification
✅ Temporal smoothing for stable predictions
✅ Confidence score for each prediction
✅ Aggregated classroom analytics (JSON output)
✅ Config-driven deployment (no hardcoded parameters)
✅ Edge-friendly and privacy-conscious (no face storage)

---

## 🧠 Engagement Signals Used

The system fuses multiple behavioral cues:

| Signal              | Description                            |
| ------------------- | -------------------------------------- |
| 👁 Eye Aspect Ratio | Detects eye closure, drowsiness, sleep |
| 👄 Mouth Ratio      | Detects yawning / fatigue              |
| 🧭 Head Roll Angle  | Detects sideways distraction           |
| 🧍 Body Pose        | Head-down posture, slouching, leaning  |
| 📱 Phone Detection  | Detects mobile phone near student      |
| ⏱ Temporal Tracking | Persistent states over time            |
| 📊 Confidence Score | Reliability of prediction              |

---

## 🏷 Engagement Labels

The model classifies each student into:

* `ATTENTIVE`
* `FOCUSED`
* `INACTIVE`
* `DROWSY`
* `SLEEPING`
* `HEAD_DOWN`
* `PHONE_DISTRACTED`
* `BORED`
* `DISTRACTED`

Active labels:

* `ATTENTIVE`
* `FOCUSED`

---

## 🏗 System Architecture

```
Camera / Video Stream
        │
        ▼
FaceMesh (Eyes, Mouth, Head Pose)
        │
YOLOv8 Pose (Body Posture)
        │
YOLO Object Detection (Phone)
        │
Feature Fusion + Temporal Smoothing
        │
Rule-based Engagement Classifier
        │
Confidence Scoring
        │
JSON Analytics Output + Visualization
```

---

## 📂 Project Structure

```
classroom-engagement-analyzer/
│
├── src/
│   ├── main.py
│   ├── engagement_logic.py
│   ├── face_utils.py
│   ├── pose_utils.py
│
├── configs/
│   └── config.yaml
│
├── models/              (ignored in git)
│
├── outputs/             (ignored in git)
│
├── samples/
│   └── demo.mp4
│
├── requirements.txt
├── .gitignore
└── README.md
```

> ⚠️ Model weights and output files are excluded from GitHub using `.gitignore`.

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/GitNinja11/classroom-engagement-analyzer.git
cd classroom-engagement-analyzer
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate     # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download YOLO Models

Place these files inside the `models/` folder:

* `yolov8n.pt`
* `yolov8n-pose.pt`

Download from: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

---

---

## ▶️ Running the Application

### ▶ Live Camera Mode

```bash
python src/main.py
```

---

### ▶ Recording Mode

```bash
python src/main.py --mode recording --video_file samples/demo.mp4
```

---

### ▶ Using Custom Config

```bash
python src/main.py --config configs/config.yaml
```

---

---

## 📊 Output

Engagement analytics are stored as JSON in:

```
outputs/
```

Example fields:

```json
{
  "timestamp": "2026-01-08 20:15:10",
  "total_students": 1.0,
  "active_pct": 100.0,
  "inactive_pct": 0.0,
  "sleeping_pct": 0.0,
  "drowsy_pct": 0.0,
  "avg_confidence": 0.87
}
```

---

---

## 🎥 Demo

📌 Demo video: *(Add your Drive / YouTube link here)*
📌 Screenshots available in `/samples`

---

---

## 🛠 Tech Stack

* **Python**
* **OpenCV**
* **MediaPipe**
* **YOLOv8 (Ultralytics)**
* **NumPy**
* **PyYAML**
* **Computer Vision**
* **Real-Time Inference**

---

## 👨‍💻 Author

**Vaishnavi Newalkar**
B.Tech ECE (IoT) — IIIT Nagpur
GitHub: [https://github.com/GitNinja11](https://github.com/GitNinja11)


