from typing import Dict
import numpy as np


def extract_yolo_pose_features(kps_xy: np.ndarray) -> Dict[str, bool]:
    """
    Extract simple pose features from YOLOv8 pose keypoints.

    COCO order (0-based):
    0:nose, 5:left_shoulder, 6:right_shoulder,
    11:left_hip, 12:right_hip,
    9:left_wrist, 10:right_wrist
    """
    if kps_xy.shape[0] < 13:
        return {
            "head_down": False,
            "torso_lean_forward": False,
            "hands_near_lap": False,
        }

    nose = kps_xy[0]
    left_sh = kps_xy[5]
    right_sh = kps_xy[6]
    left_hip = kps_xy[11]
    right_hip = kps_xy[12]
    left_wrist = kps_xy[9]
    right_wrist = kps_xy[10]

    shoulder_y = (left_sh[1] + right_sh[1]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0
    torso_height = max(hip_y - shoulder_y, 1.0)

    # --- tuneable thresholds ---
    # how far nose is below shoulders (0.0 = at shoulder, 1.0 = at hip)
    nose_drop_ratio = (nose[1] - shoulder_y) / torso_height

    # 2) Head down detection
    HEAD_DOWN_THRESH = 0.5  # nose drop ratio threshold
    head_down = nose_drop_ratio > HEAD_DOWN_THRESH

    # 3) Torso lean forward detection
    LEAN_FORWARD_THRESH = -0.2  # nose ratio threshold
    torso_lean_forward = nose_drop_ratio < LEAN_FORWARD_THRESH

    # 4) Hands near lap / phone region:
    wrists_y_avg = (left_wrist[1] + right_wrist[1]) / 2.0
    # if wrists are at or slightly below hip level → likely in lap/phone area
    HANDS_LAP_OFFSET = 0.05  # smaller = more sensitive
    hands_near_lap = wrists_y_avg > (hip_y - HANDS_LAP_OFFSET * torso_height)

    # 5) Slumped posture (bored)
    #    Shoulders relatively close to hips (short torso),
    #    not clearly head-down and not leaning forward.
    nose_y = nose[1]
    upper_body = max(hip_y - nose_y, 1.0)
    slump_ratio = torso_height / upper_body
    SLUMP_THRESH = 0.55
    slumped_posture = (
        slump_ratio < SLUMP_THRESH
        and not head_down
        and not torso_lean_forward
    )

    # 5) Phone suspected: strong condition based on pose
    phone_suspected = head_down and hands_near_lap

    return {
        "head_down": bool(head_down),
        "torso_lean_forward": bool(torso_lean_forward),
        "hands_near_lap": bool(hands_near_lap),
        "slumped_posture": bool(slumped_posture),
        "phone_suspected": bool(phone_suspected),
    }
