"""
classifier/postprocess.py

Maps CloudClassifier.predict() output to the FrameClass enum value and
the payload state fields consumed by subsystems/payload.py.

Threshold mapping (cloud_prob → FrameClass):
    >= 0.75   → CLOUDY
    >= 0.35   → PARTLY_CLOUDY
    < 0.35 and usefulness >= 0.80  → HIGH_VALUE_CLEAR
    otherwise → CLEAR
"""

from __future__ import annotations

from typing import Dict, Any

from models.enums import FrameClass


def postprocess(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map raw CloudClassifier output to payload state fields.

    Args:
        prediction: dict returned by CloudClassifier.predict()

    Returns dict with keys:
        frame_class                FrameClass enum value
        current_frame_cloud_prob   float  [0, 1]
        current_frame_usefulness   float  [0, 1]
        classifier_confidence      float  [0.5, 1.0]
        classifier_success         bool
        classifier_last_latency_s  float  (seconds)
    """
    cloud_prob = float(prediction["current_frame_cloud_prob"])
    usefulness = float(prediction["current_frame_usefulness"])
    confidence = float(prediction["classifier_confidence"])

    # ── Map continuous cloud probability → FrameClass ─────────────────────
    if cloud_prob >= 0.75:
        frame_class = FrameClass.CLOUDY
    elif cloud_prob >= 0.35:
        frame_class = FrameClass.PARTLY_CLOUDY
    elif usefulness >= 0.80:
        frame_class = FrameClass.HIGH_VALUE_CLEAR
    else:
        frame_class = FrameClass.CLEAR

    return {
        "frame_class":                  frame_class,
        "current_frame_cloud_prob":     cloud_prob,
        "current_frame_usefulness":     usefulness,
        "classifier_confidence":        confidence,
        "classifier_success":           bool(prediction.get("classifier_success", True)),
        "classifier_last_latency_s":    float(prediction.get("classifier_last_latency_s", 0.0)),
    }
