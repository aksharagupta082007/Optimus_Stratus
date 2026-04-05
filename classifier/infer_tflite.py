"""
On Apollo4 Lite this runs as a TFLite Micro C program.
This Python version is used for simulation inside the RL env.
"""
import time
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

class CloudClassifier:
    def __init__(self, model_path: str = 'classifier/model.tflite'):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype    = self.input_details[0]['dtype']   # int8 after quant
        self.input_scale, self.input_zero_point = \
            self.input_details[0]['quantization']

    def predict(self, preprocessed_frame: np.ndarray) -> dict:
        """
        preprocessed_frame: (1, 96, 96, 3) float32 in [0,1]
        Returns: {'label': 'cloudy'|'non-cloudy', 'confidence': float, 'raw_score': float}
        """
        inp = preprocessed_frame
        # Quantize float32 → int8 if model is quantized
        if self.input_dtype == np.int8:
            inp = (inp / self.input_scale + self.input_zero_point).astype(np.int8)

        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        t0 = time.perf_counter()
        self.interpreter.invoke()
        latency_s = time.perf_counter() - t0
        raw = self.interpreter.get_tensor(self.output_details[0]['index'])  

        # Dequantize int8 output → float
        out_scale, out_zp = self.output_details[0]['quantization']
        score = float((raw[0][0].astype(np.float32) - out_zp) * out_scale)

        return {
            'is_cloudy':                    score > 0.5,
            'classifier_confidence':        score if score > 0.5 else 1.0 - score,
            'current_frame_cloud_prob':     float(score),
            'current_frame_usefulness':     1.0 - float(score),   # simple inverse — clear = useful
            'classifier_last_latency_s':    0.0,                  # updated after invoke timing
            'classifier_success':           True,
            'raw_score':                    float(score),
        }   