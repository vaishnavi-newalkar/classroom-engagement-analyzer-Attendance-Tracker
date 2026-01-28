# 📊 Classroom Engagement Analyzer & Attendance Tracker

An **AI-powered classroom monitoring system** that analyzes **student engagement, behavior, and attendance** in real time using **computer vision and deep learning**.
The system integrates **face recognition, pose estimation, phone detection, and engagement classification** to generate actionable classroom analytics.

---

## 🚀 Features

### 🎯 Engagement Analysis

* Real-time **Active / Inactive classification**
* Detects:

  * 💤 Sleeping
  * 📱 Phone usage (suspected)
  * 😴 Drowsiness / Yawning
  * ⬇️ Head-down behavior
* Confidence-weighted engagement scoring

### 🧑‍🎓 Attendance Tracking

* Face recognition–based **student identity**
* Automatic **attendance marking**
* Per-student engagement statistics over time

### 🧍 Posture & Behavior Detection

* YOLOv8 **pose estimation**
* Detects:

  * Head-down posture
  * Slumped posture
  * Hands near lap
* Links pose data to individual students

### 📈 Analytics & Reporting

* Class-level engagement percentages
* Per-student engagement summaries
* JSON-based analytics output
* Optional **API integration** for backend dashboards

### 🎥 Flexible Input Modes

* Live webcam
* RTSP / IP camera streams
* Offline recorded video files

---

## 🧠 Tech Stack

| Component         | Technology               |
| ----------------- | ------------------------ |
| Language          | Python                   |
| Face Analysis     | MediaPipe FaceMesh       |
| Object Detection  | YOLOv8 (Ultralytics)     |
| Pose Estimation   | YOLOv8 Pose              |
| Face Recognition  | ArcFace-based recognizer |
| Vision            | OpenCV                   |
| ML Framework      | PyTorch                  |
| API Communication | REST (requests)          |
| Data Output       | JSON                     |

---

## 📂 Project Structure

```
ENGAGEMENT_ANALYSIS/
│
├── src/
│   ├── main.py                  # Main execution pipeline
│   ├── engagement_logic.py      # Engagement classification logic
│   ├── face_utils.py            # Eye, mouth, head angle utilities
│   ├── pose_utils.py            # YOLO pose feature extraction
│
├── face_identity/
│   └── face_recognizer.py       # Student face recognition
│
├── attendance/
│   └── attendance_tracker.py    # Attendance management
│
├── analytics/
│   └── student_metrics.py       # Per-student engagement analytics
│
├── configs/
│   └── config.yaml              # System configuration
│
├── models/
│   ├── yolov8n.pt
│   └── yolov8n-pose.pt
│
├── outputs/
│   ├── engagement_<class_id>.json
│   └── student_engagement_summary.json
│
└── requirements.txt
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download models

Place YOLO models inside the `models/` folder:

* `yolov8n.pt`
* `yolov8n-pose.pt`

---

## ▶️ Usage

### 🔴 Live Webcam Mode

```bash
python src/main.py --mode live
```

### 📡 Live RTSP / IP Camera

```bash
python src/main.py --mode live --camera_source rtsp://<camera_url>
```

### 🧠 Server-Based Camera Fetch

```bash
python src/main.py --use_server --classroom_id CSE_A1
```

### 🎥 Offline Video Analysis

```bash
python src/main.py --mode recording --video_file path/to/video.mp4
```

---

## 📤 Output

### 📁 Class-Level Engagement

Saved as:

```
outputs/engagement_<class_unique_id>.json
```

Contains:

* Active / inactive percentages
* Sleeping, drowsy, head-down stats
* Timestamped analytics

### 📁 Student-Level Summary

```
outputs/student_engagement_summary.json
```

Contains:

* Attendance status
* Engagement breakdown per student

---

## 🔗 API Integration (Optional)

The system can push engagement analytics to a backend service:

Configured in:

```yaml
configs/config.yaml
```

```yaml
api:
  server_base_url: http://localhost:8000
  engagement_api_url: http://localhost:8000/api/engagement
```

---

## 🧪 Performance Notes

* GPU acceleration supported (CUDA)
* YOLO inference optimized via frame intervals
* Smoothing applied for stable predictions

---

## 🏫 Use Cases

* Smart classrooms
* Online & hybrid learning monitoring
* Academic research on student engagement
* Attendance automation systems

---

## 🔮 Future Enhancements

* Emotion recognition
* Dashboard visualization
* Multi-classroom aggregation
* Cloud deployment
* Real-time alerts

