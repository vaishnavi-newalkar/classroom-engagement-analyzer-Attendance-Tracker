import math
from typing import List, Tuple

import numpy as np

Point = Tuple[float, float]


def euclidean_distance(p1: Point, p2: Point) -> float:
    return math.dist(p1, p2)


def eye_aspect_ratio(pts: List[Point], idxs: List[int]) -> float:
    """
    EAR using 6 landmarks: [p1, p2, p3, p4, p5, p6]
    """
    p1 = np.array(pts[idxs[0]])
    p2 = np.array(pts[idxs[1]])
    p3 = np.array(pts[idxs[2]])
    p4 = np.array(pts[idxs[3]])
    p5 = np.array(pts[idxs[4]])
    p6 = np.array(pts[idxs[5]])

    num = euclidean_distance(p2, p6) + euclidean_distance(p3, p5)
    den = 2.0 * euclidean_distance(p1, p4) + 1e-6
    return float(num / den)


def mouth_open_ratio(
    pts: List[Point],
    top_idx: int,
    bottom_idx: int,
    left_idx: int,
    right_idx: int,
) -> float:
    top = np.array(pts[top_idx])
    bottom = np.array(pts[bottom_idx])
    left = np.array(pts[left_idx])
    right = np.array(pts[right_idx])

    vertical = euclidean_distance(top, bottom)
    horizontal = euclidean_distance(left, right) + 1e-6
    return float(vertical / horizontal)


def head_roll_angle(
    pts: List[Point],
    left_eye_outer_idx: int,
    right_eye_outer_idx: int,
) -> float:
    """
    Rough roll angle: positive if right eye is lower than left eye.
    """
    left_eye = np.array(pts[left_eye_outer_idx])
    right_eye = np.array(pts[right_eye_outer_idx])

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0] + 1e-6
    angle_rad = math.atan2(dy, dx)
    return float(math.degrees(angle_rad))
