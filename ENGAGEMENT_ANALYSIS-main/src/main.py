import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import yaml
import argparse
import requests
import json
import time
from datetime import datetime
from typing import Optional
import torch
import logging
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from face_utils import eye_aspect_ratio, mouth_open_ratio, head_roll_angle
from pose_utils import extract_yolo_pose_features
from engagement_logic import classify_engagement, is_active_label
from face_identity.face_recognizer import FaceRecognizer
from attendance.attendance_tracker import AttendanceTracker
from analytics.student_metrics import StudentMetrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

FRAME_W = cfg["video"]["frame_width"]
FRAME_H = cfg["video"]["frame_height"]
MAX_FACES = cfg["video"]["max_faces"]

YOLO_INTERVAL = cfg["models"]["yolo_pose_interval"]
PHONE_INTERVAL = cfg["models"]["phone_detection_interval"]

LOG_INTERVAL_SEC = cfg["logging"]["aggregation_interval_sec"]

SERVER_BASE_URL = cfg["api"]["server_base_url"]
ENGAGEMENT_API_URL = cfg["api"]["engagement_api_url"]

CAMERA_SOURCE = None
CLASS_UNIQUE_ID = None
CAMERA_IP = None

# =========================
# LOAD MODELS (ONCE)
# =========================
logging.info("Loading YOLO models...")

pose_model = YOLO(os.path.join(MODEL_DIR, "yolov8n-pose.pt")).to(DEVICE)
phone_model = YOLO(os.path.join(MODEL_DIR, "yolov8n.pt")).to(DEVICE)


logging.info("Loading Face Recognizer...")
face_recognizer = FaceRecognizer()   # 🔴 THIS WAS MISSING

attendance_tracker = AttendanceTracker()
student_metrics = StudentMetrics()

# =========================
# PHONE CLASS
# =========================
PHONE_CLASS_ID = None
for cid, name in phone_model.names.items():
    if "phone" in name.lower():
        PHONE_CLASS_ID = cid
        logging.info(f"Phone class id = {PHONE_CLASS_ID} ({name})")
        break

if PHONE_CLASS_ID is None:
    PHONE_CLASS_ID = 67
    logging.warning("Phone class not found, using fallback 67")

PHONE_IOU_THRESH = 0.05


CLASS_UNIQUE_ID: str | None = None
CAMERA_IP: str | None = None

def box_iou(a, b) -> float:
    """
    Compute IoU between two boxes a, b = [x1, y1, x2, y2].
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area + 1e-6

    return float(inter_area / union)

# =========================
# SERVER FETCH
# =========================
def fetch_camera_source_from_server(classroom_id: str) -> object:
    """
    Ask backend server for stream URL + unique ID.
    Expected backend response (example):
    {
        "classroom_id": "CSE_A1",
        "unique_id": "abc123",
        "stream_url": "rtsp://...",
        "camera_ip": "192.168.1.50"
    }
    """
    global CLASS_UNIQUE_ID, CAMERA_IP

    url = f"{SERVER_BASE_URL}/api/cameras/{classroom_id}"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        stream_url = data.get("stream_url") or data.get("url") or data.get("camera_ip")
        CLASS_UNIQUE_ID = data.get("unique_id") or data.get("id") or classroom_id
        CAMERA_IP = data.get("camera_ip") or data.get("ip")

        if not stream_url:
            print("⚠ No stream_url from server → using webcam(0)")
            return 0

        print(f"✅ Server camera for {classroom_id}: {stream_url}")
        print(f"   unique_id={CLASS_UNIQUE_ID}, camera_ip={CAMERA_IP}")

        return stream_url

    except Exception as e:
        print(f"⚠ Server error: {e} → fallback to webcam(0)")
        CLASS_UNIQUE_ID = classroom_id
        CAMERA_IP = None
        return 0


# =========================
# ARGUMENT PARSER
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Classroom Engagement Analyzer")

    parser.add_argument("--classroom_id", type=str, default="CSE_DEFAULT")

    # For LIVE: webcam index or rtsp url
    parser.add_argument("--camera_source", type=str, default=None)

    # For OFFLINE: explicit video file path
    parser.add_argument("--video_file", type=str, default=None)

    # Use server to fetch RTSP url in live mode
    parser.add_argument("--use_server", action="store_true")

    # Optional: explicit mode
    parser.add_argument("--mode", type=str, choices=["live", "recording"],
                        default="live")

    return parser.parse_args()



# =========================
# FACE MESH INDEX CONSTANTS
# =========================
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263


# =========================
# API-B SENDER
# =========================
def send_engagement_record_to_api(record: dict) -> None:
    """
    Send one aggregated engagement record to the analytics API.
    If API is down, just print a warning and continue.
    """
    if not ENGAGEMENT_API_URL:
        return

    payload = {
        "classroom_id": CLASSROOM_ID,
        "class_unique_id": CLASS_UNIQUE_ID or CLASSROOM_ID,
        "camera_source": str(CAMERA_SOURCE),
        "camera_ip": CAMERA_IP,
        "log_interval_sec": LOG_INTERVAL_SEC,
        **record,
    }

    try:
        resp = requests.post(ENGAGEMENT_API_URL, json=payload, timeout=5)
        # print("Engagement API resp:", resp.status_code, resp.text)
    except Exception as e:
        print(f"⚠ Error sending engagement to API: {e}")


# =========================
# MAIN
# =========================
def main():
    global CAMERA_SOURCE, CLASSROOM_ID, CLASS_UNIQUE_ID

    args = parse_args()
    CLASSROOM_ID = args.classroom_id

    # -----------------------
    # Smarter Camera Selection Logic (LIVE + RECORDING)
    # -----------------------
    if args.mode == "recording":
        # OFFLINE mode: must have a file path
        if args.video_file is None:
            print("❌ recording mode requires --video_file")
            return
        CAMERA_SOURCE = args.video_file
        CLASS_UNIQUE_ID = f"{CLASSROOM_ID}_RECORDING"
        print(f"🎥 [RECORDING MODE] Using video file: {CAMERA_SOURCE}")

    else:  # LIVE mode
        if args.camera_source is not None:
            # explicit source: rtsp url or webcam index
            try:
                CAMERA_SOURCE = int(args.camera_source)
            except ValueError:
                CAMERA_SOURCE = args.camera_source
            CLASS_UNIQUE_ID = CLASSROOM_ID
            print(f"🎥 [LIVE MODE] Using explicit camera source: {CAMERA_SOURCE}")

        elif args.use_server:
            # fetch rtsp/url from backend using classroom_id
            CAMERA_SOURCE = fetch_camera_source_from_server(CLASSROOM_ID)
            print(f"🎥 [LIVE MODE] Using server-provided source: {CAMERA_SOURCE}")

        else:
            # fallback webcam(0)
            CAMERA_SOURCE = 0
            CLASS_UNIQUE_ID = CLASSROOM_ID
            print("🎥 [LIVE MODE] No source provided → using webcam(0)")

    # -----------------------
    # Engagement record list
    # -----------------------
    engagement_records = []

    # =========================
    # Per-student identity + analytics
    # =========================
    face_recognizer = FaceRecognizer()
    attendance_tracker = AttendanceTracker()
    student_metrics = StudentMetrics()

    # Frame smoothing vars
    # Frame smoothing vars
    closed_eye_frames = [0] * MAX_FACES
    head_down_frames = [0] * MAX_FACES  # NEW: track how long head is down

    SMOOTH_ALPHA = 0.7
    EAR_smooth = [0.0] * MAX_FACES
    MOUTH_smooth = [0.0] * MAX_FACES
    ROLL_smooth = [0.0] * MAX_FACES


    # Interval accumulators
    last_log_time = time.time()
    agg_samples = 0
    sum_students = 0.0
    sum_active = 0.0
    sum_inactive = 0.0
    sum_confidence = 0.0

    # New: interval sums for sleeping / head-down / drowsy
    sum_sleeping = 0.0
    sum_head_down = 0.0
    sum_drowsy = 0.0

    # YOLO cache
    last_yolo_time = 0.0
    last_yolo_kps_all = None
    last_yolo_boxes = None
    last_yolo_features = []

    # Phone detection cache
    last_phone_time = 0.0
    last_phone_boxes = []  # list of [x1, y1, x2, y2] for detected phones


    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAMERA_SOURCE)

    with mp_face.FaceMesh(
        max_num_faces=MAX_FACES,
        refine_landmarks=True,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            # Keep a copy of the original frame for phone detection
            raw_frame = frame.copy()

            raw_h, raw_w = raw_frame.shape[:2]

            # If portrait video, rotate it to landscape
            if raw_h > raw_w:
                raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_CLOCKWISE)
                raw_h, raw_w = raw_w, raw_h  # swap after rotation


            # This resized frame is used for FaceMesh + pose
            frame = cv2.resize(raw_frame, (FRAME_W, FRAME_H))

            h, w, _ = frame.shape
            now = time.time()

            # -----------------------
            # Face recognition (ONCE per frame)
            # -----------------------
            recognized_faces = face_recognizer.detect_and_identify(frame)


            # -----------------------
            # YOLO pose (1 FPS)
            # -----------------------
            yolo_kps = last_yolo_kps_all
            yolo_boxes = last_yolo_boxes
            yolo_feats = last_yolo_features

            if now - last_yolo_time >= YOLO_INTERVAL:
                results = pose_model(frame, imgsz=256, verbose=False)[0]

                yolo_kps = None
                yolo_boxes = None
                yolo_feats = []

                if results.keypoints is not None:
                    yolo_kps = results.keypoints.xy.cpu().numpy()
                    yolo_boxes = results.boxes.xyxy.cpu().numpy()

                    for kps in yolo_kps:
                        feats = extract_yolo_pose_features(kps)
                        yolo_feats.append(feats)

                last_yolo_kps_all = yolo_kps
                last_yolo_boxes = yolo_boxes
                last_yolo_features = yolo_feats
                last_yolo_time = now

            # -----------------------
            # Phone detection (object YOLO) on FULL-RES frame
            # -----------------------
            phone_boxes = last_phone_boxes

            if now - last_phone_time >= PHONE_INTERVAL:
                phone_boxes = []
                det_results = phone_model(raw_frame, imgsz=640, verbose=False)[0]

                if det_results.boxes is not None:
                    det_xyxy = det_results.boxes.xyxy.cpu().numpy()
                    det_cls = det_results.boxes.cls.cpu().numpy()

                    # DEBUG: show all detected classes this cycle
                    print(
                        "[YOLO-PHONE] classes:",
                        [f"{int(c)}:{phone_model.names[int(c)]}" for c in det_cls],
                    )

                    # scale factors: raw frame -> resized frame
                    scale_x = FRAME_W / raw_w
                    scale_y = FRAME_H / raw_h

                    for box, cls_id in zip(det_xyxy, det_cls):
                        cls_id_int = int(cls_id)
                        x1_raw, y1_raw, x2_raw, y2_raw = box

                        # map detection coordinates to resized frame
                        x1 = int(x1_raw * scale_x)
                        y1 = int(y1_raw * scale_y)
                        x2 = int(x2_raw * scale_x)
                        y2 = int(y2_raw * scale_y)

                        # draw ALL detections with their class name (for debugging)
                        class_name = phone_model.names.get(cls_id_int, str(cls_id_int))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(
                            frame,
                            f"{class_name}({cls_id_int})",
                            (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )

                        # store only the PHONE_CLASS_ID boxes in phone_boxes
                        if cls_id_int == PHONE_CLASS_ID:
                            phone_boxes.append([x1, y1, x2, y2])

                print("[YOLO-PHONE] phone_boxes:", len(phone_boxes))
                last_phone_boxes = phone_boxes
                last_phone_time = now

            # Draw phone boxes (subset) in red and label PHONE
            for (px1, py1, px2, py2) in phone_boxes:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "PHONE",
                    (px1, max(py1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )



            # -----------------------
            # FaceMesh
            # -----------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = face_mesh.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            total_faces = 0
            active_count = 0
            inactive_count = 0
            
            # New: per-frame class-level counters
            sleeping_count = 0
            head_down_count = 0
            drowsy_count = 0    # includes yawning


            if res.multi_face_landmarks:
                for i, face_lms in enumerate(res.multi_face_landmarks):
                    if i >= MAX_FACES:
                        break

                    pts = [(lm.x * w, lm.y * h) for lm in face_lms.landmark]

                    left_ear = eye_aspect_ratio(pts, LEFT_EYE_IDXS)
                    right_ear = eye_aspect_ratio(pts, RIGHT_EYE_IDXS)
                    avg_ear = (left_ear + right_ear) / 2.0

                    mouth_r = mouth_open_ratio(
                        pts, MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX,
                        MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX
                    )

                    roll_deg = head_roll_angle(pts, LEFT_EYE_OUTER_IDX, RIGHT_EYE_OUTER_IDX)

                    # smoothing
                    EAR_smooth[i] = SMOOTH_ALPHA * EAR_smooth[i] + (1-SMOOTH_ALPHA)*avg_ear
                    MOUTH_smooth[i] = SMOOTH_ALPHA * MOUTH_smooth[i] + (1-SMOOTH_ALPHA)*mouth_r
                    ROLL_smooth[i] = SMOOTH_ALPHA * ROLL_smooth[i] + (1-SMOOTH_ALPHA)*roll_deg

                    ear_final = EAR_smooth[i]
                    mouth_final = MOUTH_smooth[i]
                    roll_final = ROLL_smooth[i]

                    # bbox + center
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))

 

                    cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
                    # Safe face ROI extraction
                    x1 = max(0, x_min)
                    y1 = max(0, y_min)
                    x2 = min(w, x_max)
                    y2 = min(h, y_max)

                    # ---- PAD FACE ROI FOR ARCFACE ----
                    pad_x = int(0.25 * (x2 - x1))
                    pad_y = int(0.25 * (y2 - y1))

                    px1 = max(0, x1 - pad_x)
                    py1 = max(0, y1 - pad_y)
                    px2 = min(w, x2 + pad_x)
                    py2 = min(h, y2 + pad_y)
                    

                    print("Recognized:", [(r["id"], r["bbox"]) for r in recognized_faces])

                    # -----------------------
                    # Attach identity using IoU
                    # -----------------------
                    student_id = "unknown"

                    face_bbox = [px1, py1, px2, py2]


                    for rf in recognized_faces:
                        rx1, ry1, rx2, ry2 = rf["bbox"]
                        rec_bbox = [rx1, ry1, rx2, ry2]

                        iou = box_iou(face_bbox, rec_bbox)

                        if iou > 0.25:   # key threshold
                            student_id = rf["id"]
                            print("[FACE-ID]", student_id)
                            break



                    # Match to YOLO person
                    matched_pose = None
                    matched_box = None  # YOLO person box for this face

                    if last_yolo_kps_all is not None:
                        best = None
                        best_d = 1e9
                        for idx, kps in enumerate(last_yolo_kps_all):
                            nose = kps[0]
                            d = (nose[0] - cx) ** 2 + (nose[1] - cy) ** 2
                            if d < best_d:
                                best_d = d
                                best = idx

                        if best is not None and best < len(last_yolo_features):
                            matched_pose = last_yolo_features[best]
                            if last_yolo_boxes is not None and best < len(last_yolo_boxes):
                                matched_box = last_yolo_boxes[best].astype(int)

                    # -----------------------
                    # Attach phone detection to this student
                    # -----------------------
                    if matched_pose is not None and matched_box is not None:
                        has_phone_near = False
                        for pbox in phone_boxes:
                            iou_val = box_iou(matched_box, pbox)
                            # DEBUG: see how much overlap we get
                            print("[IOU] person-phone IoU:", iou_val)
                            if iou_val > PHONE_IOU_THRESH:
                                has_phone_near = True
                                break

                        if has_phone_near:
                            print("[PHONE] Phone attached to this student!")  # DEBUG
                            matched_pose["phone_suspected"] = True


                    # Eye closure counting (for sleeping)
                    if ear_final < 0.18:
                        closed_eye_frames[i] += 1
                    else:
                        closed_eye_frames[i] = 0

                    # Head-down duration counting (for "looking down for long" / phone)
                    if matched_pose and matched_pose.get("head_down", False):
                        head_down_frames[i] += 1
                    else:
                        head_down_frames[i] = 0
     



                    # final engagement label
                                       # final engagement label
                    label, confidence = classify_engagement(
                        ear_final,
                        mouth_final,
                        roll_final,
                        closed_eye_frames[i],
                        pose_features=matched_pose,
                        fps=15.0,
                        head_down_frames=head_down_frames[i],
                    )

                    # Attendance + per-student engagement tracking
                    if student_id != "unknown":
                        attendance_tracker.mark(student_id)
                        student_metrics.update(student_id, label, confidence)



                    total_faces += 1
                    if is_active_label(label):
                        active_count += 1
                    else:
                        inactive_count += 1
                    
                    sum_confidence += confidence

                    # =========================
                    # New: sleeping / head-down / drowsy counts
                    # Adjust the label checks to match your actual labels
                    # (e.g., "SLEEPING", "HEAD_DOWN", "DROWSY", "YAWNING")
                    # =========================
                    low = label.lower()

                    # sleeping → eyes closed for long
                    if "sleep" in low:          # e.g., "SLEEPING"
                        sleeping_count += 1

                    # clear head-down detection from label or pose features
                    if "head down" in low or (matched_pose and matched_pose.get("head_down", False)):
                        head_down_count += 1

                    # drowsy / yawning
                    if "drowsy" in low or "yawn" in low:
                        drowsy_count += 1


                    # drawing: FaceMesh + bbox + label
                    mp_drawing.draw_landmarks(
                        frame, face_lms,
                        mp_face.FACEMESH_TESSELATION,
                        None, mp_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        frame, face_lms,
                        mp_face.FACEMESH_CONTOURS,
                        None, mp_styles.get_default_face_mesh_contours_style(),
                    )
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 1)
                    display_text = f"{student_id} | {label}"
                    cv2.putText(frame, display_text, (x_min, y_min-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)


            # -----------------------
            # Draw YOLO body box + dynamic pose label
            # -----------------------
            if last_yolo_boxes is not None:
                for idx, box in enumerate(last_yolo_boxes):
                    x1, y1, x2, y2 = box.astype(int)

                    # default label
                    pose_label = "POSE"

                    if last_yolo_features and idx < len(last_yolo_features):
                        feats = last_yolo_features[idx]

                        if feats.get("phone_suspected"):
                            pose_label = "PHONE SUSPECTED"
                        elif feats.get("head_down"):
                            pose_label = "HEAD DOWN"
                        elif feats.get("slumped_posture"):
                            pose_label = "SLUMPED"
                        elif feats.get("hands_near_lap"):
                            pose_label = "HANDS NEAR LAP"
                        else:
                            pose_label = "ATTENTIVE POSTURE"


                    # body rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    # label text above box
                    cv2.putText(
                        frame,
                        f"POSE: {pose_label}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

            # -----------------------
            # Class-level summary
            # -----------------------
            if total_faces > 0:
                students = total_faces
            elif last_yolo_features:
                students = len(last_yolo_features)
                active_count = sum(1 for f in last_yolo_features
                                   if not (f["head_down"] or f["hands_near_lap"]))
                inactive_count = len(last_yolo_features) - active_count
            else:
                students = active_count = inactive_count = 0

            active_pct = (active_count / students * 100) if students > 0 else 0.0
            inactive_pct = (inactive_count / students * 100) if students > 0 else 0.0

            # accumulate
            agg_samples += 1
            sum_students += students
            sum_active += active_count
            sum_inactive += inactive_count

            # New: accumulate sleeping / head-down / drowsy counts
            sum_sleeping += sleeping_count
            sum_head_down += head_down_count
            sum_drowsy += drowsy_count


            # interval logging
            current = time.time()
            if current - last_log_time >= LOG_INTERVAL_SEC and agg_samples > 0:

                if sum_students > 0:
                    avg_students = sum_students / agg_samples
                    avg_active = sum_active / agg_samples
                    avg_inactive = sum_inactive / agg_samples

                    pct_active = (sum_active / sum_students) * 100
                    pct_inactive = (sum_inactive / sum_students) * 100

                    # New: interval percentages
                    pct_sleeping = (sum_sleeping / sum_students) * 100
                    pct_head_down = (sum_head_down / sum_students) * 100
                    pct_drowsy = (sum_drowsy / sum_students) * 100
                else:
                    avg_students = avg_active = avg_inactive = 0.0
                    pct_active = pct_inactive = 0.0
                    pct_sleeping = pct_head_down = pct_drowsy = 0.0


                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                record = {
                    "timestamp": timestamp,
                    "total_students": round(avg_students, 2),
                    "active_count": round(avg_active, 2),
                    "inactive_count": round(avg_inactive, 2),
                    "active_pct": round(pct_active, 2),
                    "inactive_pct": round(pct_inactive, 2),

                    # New fields in JSON
                    "sleeping_pct": round(pct_sleeping, 2),
                    "head_down_pct": round(pct_head_down, 2),
                    "drowsy_pct": round(pct_drowsy, 2),

                    "avg_confidence": round(sum_confidence / max(sum_students, 1), 3),

                }


                # Store in memory for local JSON
                engagement_records.append(record)

                # send to API-B
                send_engagement_record_to_api(record)

                # reset for next interval
                last_log_time = current
                agg_samples = 0
                sum_students = 0.0
                sum_active = 0.0
                sum_inactive = 0.0
                sum_sleeping = 0.0
                sum_head_down = 0.0
                sum_drowsy = 0.0
                sum_confidence = 0.0


            # UI summary
            text = f"Students: {students} | Active: {active_pct:.0f}% | Inactive: {inactive_pct:.0f}%"
            cv2.putText(frame, text, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Engagement (FaceMesh + YOLO Pose)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # ========================
    # SAVE ANALYSIS AS JSON
    # ========================
    output_id = CLASS_UNIQUE_ID or CLASSROOM_ID

    os.makedirs("outputs", exist_ok=True)

    json_filename = os.path.join(
        "outputs",
        f"engagement_{output_id}.json"
    )

    summary = {
        "classroom_id": CLASSROOM_ID,
        "class_unique_id": output_id,
        "camera_source": str(CAMERA_SOURCE),
        "camera_ip": CAMERA_IP,
        "log_interval_sec": LOG_INTERVAL_SEC,
        "records": engagement_records,
    }

    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Saved engagement JSON → {json_filename}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

    # ========================
    # Save per-student engagement summary
    # ========================
    student_summary = student_metrics.summary()

    summary_path = os.path.join("outputs", "student_engagement_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(student_summary, f, indent=2)

    print(f"💾 Saved per-student summary → {summary_path}")


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
