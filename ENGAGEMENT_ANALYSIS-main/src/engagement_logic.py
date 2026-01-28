from typing import Dict

# =========================
# LABEL CONSTANTS
# =========================

LABEL_ATTENTIVE = "ATTENTIVE"
LABEL_FOCUSED = "FOCUSED"
LABEL_INACTIVE = "INACTIVE"
LABEL_DROWSY = "DROWSY/YAWNING"
LABEL_SLEEPING = "SLEEPING"
LABEL_HEAD_DOWN = "HEAD_DOWN"
LABEL_PHONE = "PHONE_DISTRACTED"
LABEL_BORED = "BORED"
LABEL_DISTRACTED = "DISTRACTED"

ACTIVE_LABELS = {
    LABEL_ATTENTIVE,
    LABEL_FOCUSED,
}


def classify_engagement(
    ear: float,
    mouth_r: float,
    roll_deg: float,
    closed_eye_frames: int,
    pose_features: Dict[str, bool] | None = None,
    fps: float = 15.0,
    head_down_frames: int = 0,
) -> tuple[str, float]:

    """
    Rule-based engagement classifier using face + pose features.
    """

    pose_features = pose_features or {}

    head_down = pose_features.get("head_down", False)
    torso_lean = pose_features.get("torso_lean_forward", False)
    hands_lap = pose_features.get("hands_near_lap", False)
    slumped = pose_features.get("slumped_posture", False)
    phone_suspected = pose_features.get("phone_suspected", False)

    # --- thresholds (tune for your classroom) ---
    EAR_DROWSY = 0.20
    EAR_SLEEP = 0.18
    MAR_YAWN = 0.55
    ROLL_DISTRACT = 20.0  # degrees
    SLEEP_FRAMES = int(2.5 * fps)  # ~2.5 seconds with eyes closed

    # "looking down for long"
    HEAD_DOWN_LONG_SEC = 3.0
    HEAD_DOWN_LONG_FRAMES = int(HEAD_DOWN_LONG_SEC * fps)

       # 1) Sleeping
    if closed_eye_frames >= SLEEP_FRAMES or (ear < EAR_SLEEP and head_down):
        return LABEL_SLEEPING,0.95

    # 2) Drowsy / yawning
    if ear < EAR_DROWSY and mouth_r > MAR_YAWN:
        return LABEL_DROWSY, 0.85

    # 5) Phone usage (pose-based)
    if phone_suspected:
            return LABEL_PHONE, 0.90


    # 6) Looking down for long (even if phone not clearly seen)
    if head_down and head_down_frames >= HEAD_DOWN_LONG_FRAMES:
        return LABEL_HEAD_DOWN, 0.85

    # Eyes open + head up → attentive (override posture)
    if ear >= EAR_DROWSY and not head_down:
        return LABEL_ATTENTIVE, 0.85


    # 3) Short-term head down / phone (generic inattentive)
    # Hands alone should NOT mark inactive
    if head_down and ear >= EAR_SLEEP:
        return LABEL_INACTIVE, 0.75


    # 4) Slumped posture (bored)
    if slumped:
        return LABEL_BORED, 0.80

    # Sideways distraction
    if abs(roll_deg) > ROLL_DISTRACT:
        return LABEL_DISTRACTED, 0.75

    # Focused lean-in
    if torso_lean and ear >= EAR_DROWSY:
        return LABEL_FOCUSED, 0.90

    # Default
    return LABEL_ATTENTIVE , 0.70



def is_active_label(label: str) -> bool:
    """
    Decide whether a label counts as 'active / attentive'
    """
    return label in ACTIVE_LABELS

