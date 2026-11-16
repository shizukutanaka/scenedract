"""Landmarks -> anatomical measurements (pure geometry, no I/O).

Input is one pose as 33 BlazePose landmarks ``(x, y, visibility)`` in
normalised image coordinates.  Output is a small dict of *derived
quantities only*; the landmark coordinates themselves are consumed by
these functions and never returned, stored or logged by callers.

What is measured (design 2026-11-15, section 2.1):

- ``proportions`` -- segment length / stature for upperArm, forearm,
  thigh, shin and torso.  Segments are joint-to-joint distances, the
  same quantities the SceneDract consumer derives from its mannequin
  part table, so learned ranges and scene figures are comparable.
  ``head`` is intentionally not measured: pose landmarks carry no
  crown or chin point, so a head length would be invented rather than
  measured.
- ``jointAngles`` -- elbow, knee, shoulder and hip angles in degrees
  (image plane; left and right sides pooled into one stream per joint).
- ``grounding`` -- ``comMarginRatio``: the signed distance from the
  centre-of-mass proxy (hip midpoint) projected to the support
  interval spanned by the feet, normalised by the COM height above
  the feet.  This is an image-plane proxy for the quantity the
  SceneDract physics audit computes in 3D; it is labelled as such.
- ``faceRelations`` -- ``eyeSpacingRatio``: inter-eye distance over
  inter-ear distance (a face-width proxy), frontal poses only.

Stature estimate (stated approximation, not hidden): the vertical
distance from the eye midpoint to the ground point (lowest foot
landmark), plus a crown offset of ``CROWN_OFFSET_FACTOR`` times the
eye-to-nose distance.  Basis: in the classical facial-thirds division
the eye-to-nose distance is roughly one third of head height, and the
crown-to-eye distance is roughly 0.53 of head height, hence a factor
of about 1.6.  Proportion statistics are only collected from near
upright, near frontal poses (gates below) so perspective distortion
stays small.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

Landmark = Tuple[float, float, float]

# BlazePose landmark indices.
NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32

MIN_VISIBILITY = 0.5
CROWN_OFFSET_FACTOR = 1.6
# Proportion gates: the shoulder->hip axis must be near vertical and
# the shoulder line near horizontal (degrees).
MAX_TORSO_TILT_DEG = 20.0
MAX_SHOULDER_ROLL_DEG = 15.0
# Frontal gate for face relations: nose must sit near the ear midpoint.
MAX_NOSE_ASYMMETRY = 0.25


def _pt(lm: Sequence[Landmark], idx: int, w: float, h: float) -> Tuple[float, float]:
    x, y, _v = lm[idx]
    return (x * w, y * h)


def _vis(lm: Sequence[Landmark], idx: int) -> float:
    return lm[idx][2]


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle(a: Tuple[float, float], vertex: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle at ``vertex`` in degrees, in [0, 180]."""
    v1 = (a[0] - vertex[0], a[1] - vertex[1])
    v2 = (c[0] - vertex[0], c[1] - vertex[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _all_visible(lm: Sequence[Landmark], idxs) -> bool:
    return all(_vis(lm, i) >= MIN_VISIBILITY for i in idxs)


def measurements_from_landmarks(
    landmarks: Sequence[Landmark], width: float, height: float
) -> Dict[str, Dict[str, object]]:
    """Compute every measurement group the pose supports.

    Groups whose gates fail are simply absent from the result; a pose
    that supports nothing returns ``{}``.
    """
    if len(landmarks) < 33 or width <= 0 or height <= 0:
        return {}
    out: Dict[str, Dict[str, float]] = {}

    def P(i):
        return _pt(landmarks, i, width, height)

    mid_shoulder = _mid(P(LEFT_SHOULDER), P(RIGHT_SHOULDER))
    mid_hip = _mid(P(LEFT_HIP), P(RIGHT_HIP))

    # ---- joint angles (per joint; each value is the list of per-side
    # samples, which the aggregator pools into one stream per joint) ----
    angles: Dict[str, list] = {}
    joint_defs = {
        "elbow": [
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        ],
        "knee": [
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        ],
        "shoulder": [
            (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
            (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
        ],
        "hip": [
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
        ],
    }
    for joint, sides in joint_defs.items():
        vals = []
        for a, v, c in sides:
            if _all_visible(landmarks, (a, v, c)):
                ang = _angle(P(a), P(v), P(c))
                if not math.isnan(ang):
                    vals.append(ang)
        if vals:
            angles[joint] = vals
    if angles:
        out["jointAngles"] = angles

    # ---- gates shared by proportions / grounding ----
    torso_axis = (mid_hip[0] - mid_shoulder[0], mid_hip[1] - mid_shoulder[1])
    torso_len_px = math.hypot(*torso_axis)
    upright = False
    if torso_len_px > 0:
        tilt = math.degrees(math.atan2(abs(torso_axis[0]), abs(torso_axis[1])))
        shoulder_roll = math.degrees(
            math.atan2(
                abs(P(LEFT_SHOULDER)[1] - P(RIGHT_SHOULDER)[1]),
                abs(P(LEFT_SHOULDER)[0] - P(RIGHT_SHOULDER)[0]) + 1e-9,
            )
        )
        upright = tilt <= MAX_TORSO_TILT_DEG and shoulder_roll <= MAX_SHOULDER_ROLL_DEG

    # ---- proportions (upright poses only) ----
    core_visible = _all_visible(
        landmarks,
        (
            LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
            LEFT_EYE, RIGHT_EYE, NOSE,
            LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL,
            LEFT_FOOT, RIGHT_FOOT,
        ),
    )
    if upright and core_visible:
        eye_mid = _mid(P(LEFT_EYE), P(RIGHT_EYE))
        ground_y = max(
            P(LEFT_HEEL)[1], P(RIGHT_HEEL)[1], P(LEFT_FOOT)[1], P(RIGHT_FOOT)[1],
            P(LEFT_ANKLE)[1], P(RIGHT_ANKLE)[1],
        )
        ground_mid = (mid_hip[0], ground_y)
        crown_offset = CROWN_OFFSET_FACTOR * _dist(eye_mid, P(NOSE))
        stature = _dist(eye_mid, ground_mid) + crown_offset
        if stature > 0:
            def seg(i, j):
                return _dist(P(i), P(j)) / stature

            def seg_mean(pairs):
                vals = [seg(i, j) for i, j in pairs if _all_visible(landmarks, (i, j))]
                return sum(vals) / len(vals) if vals else None

            props: Dict[str, float] = {}
            for name, pairs in {
                "upperArm": [(LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)],
                "forearm": [(LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST)],
                "thigh": [(LEFT_HIP, LEFT_KNEE), (RIGHT_HIP, RIGHT_KNEE)],
                "shin": [(LEFT_KNEE, LEFT_ANKLE), (RIGHT_KNEE, RIGHT_ANKLE)],
            }.items():
                v = seg_mean(pairs)
                if v is not None:
                    props[name] = v
            props["torso"] = torso_len_px / stature
            if props:
                out["proportions"] = props

            # ---- grounding (image-plane proxy) ----
            feet_x = [
                P(LEFT_HEEL)[0], P(RIGHT_HEEL)[0], P(LEFT_FOOT)[0], P(RIGHT_FOOT)[0],
                P(LEFT_ANKLE)[0], P(RIGHT_ANKLE)[0],
            ]
            lo, hi = min(feet_x), max(feet_x)
            com_x = mid_hip[0]
            com_height = ground_y - mid_hip[1]
            if com_height > 0:
                if lo <= com_x <= hi:
                    margin = min(com_x - lo, hi - com_x)
                else:
                    margin = -min(abs(com_x - lo), abs(com_x - hi))
                out["grounding"] = {"comMarginRatio": margin / com_height}

    # ---- face relations (frontal poses only) ----
    if _all_visible(landmarks, (LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR, NOSE)):
        ear_mid = _mid(P(LEFT_EAR), P(RIGHT_EAR))
        ear_span = _dist(P(LEFT_EAR), P(RIGHT_EAR))
        if ear_span > 0:
            asymmetry = abs(P(NOSE)[0] - ear_mid[0]) / ear_span
            if asymmetry <= MAX_NOSE_ASYMMETRY:
                out["faceRelations"] = {
                    "eyeSpacingRatio": _dist(P(LEFT_EYE), P(RIGHT_EYE)) / ear_span
                }

    return out
