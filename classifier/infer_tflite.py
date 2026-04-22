"""
classifier/infer_tflite.py

Wraps cloud_classifier.tflite for inference within the RL environment.

Handles:
- INT8 quantisation  (input quantise + output dequantise)
- Temperature-scaling calibration (fix overconfidence from training)

Usage:
    from classifier.preprocess import parse_tfrecord_numpy, prepare_input
    from classifier.infer_tflite import CloudClassifier

    clf = CloudClassifier()
    img, _ = parse_tfrecord_numpy(raw_bytes)   # (96, 96, 3)
    inp     = prepare_input(img)               # (1, 96, 96, 3)
    result  = clf.predict(inp)
    # result['is_cloudy']  → bool
    # result['current_frame_cloud_prob'] → float in [0, 1]
"""

from __future__ import annotations

import os
import time
import numpy as np
from typing import Dict

# ── TFLite runtime: prefer lightweight tflite_runtime, fall back to full TF ──
try:
    import tflite_runtime.interpreter as _tflite
    _Interpreter = _tflite.Interpreter
except ImportError:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter

# ── Default model path (relative to this file) ────────────────────────────────
_DEFAULT_MODEL_PATH: str = os.path.join(
    os.path.dirname(__file__),
    "cloud_classifier.tflite",
)

# ── Temperature-scaling calibration ───────────────────────────────────────────
# Obtained from Colab notebook Cell 6c / Part C output.
# T > 1 means the model was overconfident; T = 1 means no scaling.
#
# !! UPDATE this value with your actual Colab output before training the RL agent !!
# Example line in Colab output:  "Best T = 2.35"
CALIBRATION_T: float = 1.0   # <-- replace with actual value


# ─────────────────────────────────────────────────────────────────────────────

class CloudClassifier:
    """
    TFLite-based binary cloud / non-cloud image classifier.

    Thread-safety: one interpreter per instance; do not share across threads.
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TFLite model not found at '{model_path}'.\n"
                "Expected: classifier/cloud_classifier.tflite\n"
                "Fix: rename 'classifier/cloud_classifier (2).tflite' → "
                "'classifier/cloud_classifier.tflite'"
            )

        self.interpreter = _Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_dtype = self.input_details[0]["dtype"]
        quant = self.input_details[0]["quantization"]
        self.input_scale:      float = float(quant[0])
        self.input_zero_point: int   = int(quant[1])

    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, preprocessed_frame: np.ndarray) -> Dict[str, object]:
        """
        Run inference on one pre-processed frame.

        Args:
            preprocessed_frame: shape (1, 96, 96, 3) float32 in [0, 1].

        Returns a dict with:
            is_cloudy                 bool    — True if cloud_prob > 0.5
            classifier_confidence     float   — max(prob, 1-prob), range [0.5, 1.0]
            current_frame_cloud_prob  float   — calibrated cloud probability [0, 1]
            current_frame_usefulness  float   — 1 - cloud_prob [0, 1]
            classifier_last_latency_s float   — wall-clock inference time (s)
            classifier_success        bool    — always True on this path
            raw_score                 float   — uncalibrated model output
        """
        inp = preprocessed_frame.astype(np.float32)

        # ── Quantise float32 → int8 if model uses INT8 input ─────────────────
        if self.input_dtype == np.int8:
            if self.input_scale > 0:
                inp = (inp / self.input_scale + self.input_zero_point).astype(np.int8)
            else:
                inp = inp.astype(np.int8)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)

        t0 = time.perf_counter()
        self.interpreter.invoke()
        latency_s = time.perf_counter() - t0

        raw_out = self.interpreter.get_tensor(self.output_details[0]["index"])

        # ── Dequantise int8 output → float32 ─────────────────────────────────
        out_scale, out_zp = self.output_details[0]["quantization"]
        if out_scale > 0:
            raw_score = float((raw_out[0][0].astype(np.float32) - out_zp) * out_scale)
        else:
            raw_score = float(raw_out[0][0])

        raw_score = float(np.clip(raw_score, 1e-7, 1.0 - 1e-7))

        # ── Temperature-scaling calibration ──────────────────────────────────
        if CALIBRATION_T != 1.0:
            logit = float(np.log(raw_score / (1.0 - raw_score)))
            score = float(1.0 / (1.0 + np.exp(-logit / CALIBRATION_T)))
        else:
            score = raw_score

        score = float(np.clip(score, 1e-7, 1.0 - 1e-7))

        return {
            "is_cloudy":                    score > 0.5,
            "classifier_confidence":        score if score > 0.5 else 1.0 - score,
            "current_frame_cloud_prob":     score,
            "current_frame_usefulness":     1.0 - score,
            "classifier_last_latency_s":    latency_s,
            "classifier_success":           True,
            "raw_score":                    raw_score,
        }
