import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# CONFIG
# =========================
TFLITE_MODEL_PATH = "model.tflite"

# Dataset structure expected:
# dataset/
#   Cloudy/
#   NonCloudy/
DATASET_DIR = "dataset"

# Change if your model expects a different size
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Set according to your label mapping
# Here:
# Cloudy -> 0
# NonCloudy -> 1
CLASS_NAMES = ["Cloudy", "NonCloudy"]


# =========================
# LOAD TFLITE MODEL
# =========================
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]
output_dtype = output_details[0]["dtype"]

# For quantized models
input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

print(f"\nModel input shape: {input_shape}")
print(f"Model input dtype: {input_dtype}")
print(f"Model output dtype: {output_dtype}")
print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
print(f"Output quantization: scale={output_scale}, zero_point={output_zero_point}")


# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32)

    # Common preprocessing for float models
    # If your model was trained differently, adjust this line
    img = img / 255.0

    img = np.expand_dims(img, axis=0)  # shape: (1, H, W, C)

    # Handle quantized input
    if input_dtype == np.uint8 or input_dtype == np.int8:
        if input_scale > 0:
            img = img / input_scale + input_zero_point
        img = np.round(img).astype(input_dtype)
    else:
        img = img.astype(input_dtype)

    return img


# =========================
# INFERENCE
# =========================
def predict_tflite(img_path):
    input_data = preprocess_image(img_path)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)

    # Dequantize output if needed
    if output_dtype == np.uint8 or output_dtype == np.int8:
        if output_scale > 0:
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        else:
            output = output.astype(np.float32)

    output = np.array(output)

    # Handle common output formats
    # Case 1: sigmoid binary output -> shape (1,1) or similar
    if output.ndim == 2 and output.shape[1] == 1:
        score = float(output[0][0])
        pred = 1 if score >= 0.5 else 0
        return pred, score

    # Case 2: softmax/logits with 2 classes -> shape (1,2)
    elif output.ndim == 2 and output.shape[1] == 2:
        probs = output[0]
        pred = int(np.argmax(probs))
        score = float(probs[pred])
        return pred, score

    # Fallback
    else:
        flat = output.flatten()
        if len(flat) == 1:
            score = float(flat[0])
            pred = 1 if score >= 0.5 else 0
            return pred, score
        else:
            pred = int(np.argmax(flat))
            score = float(flat[pred])
            return pred, score


# =========================
# LOAD DATASET
# =========================
def collect_dataset_paths(dataset_dir):
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: folder not found -> {class_dir}")
            continue

        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(class_dir, file_name))
                labels.append(class_idx)

    return image_paths, labels


# =========================
# EVALUATE
# =========================
image_paths, y_true = collect_dataset_paths(DATASET_DIR)

if len(image_paths) == 0:
    raise ValueError("No images found. Check DATASET_DIR and folder structure.")

y_pred = []
scores = []

for i, img_path in enumerate(image_paths):
    try:
        pred, score = predict_tflite(img_path)
        y_pred.append(pred)
        scores.append(score)
    except Exception as e:
        print(f"Error on {img_path}: {e}")

print("\nTotal evaluated images:", len(y_pred))

# Metrics
cm = confusion_matrix(y_true[:len(y_pred)], y_pred)
acc = accuracy_score(y_true[:len(y_pred)], y_pred)
prec = precision_score(y_true[:len(y_pred)], y_pred, average="binary", pos_label=1, zero_division=0)
rec = recall_score(y_true[:len(y_pred)], y_pred, average="binary", pos_label=1, zero_division=0)
f1 = f1_score(y_true[:len(y_pred)], y_pred, average="binary", pos_label=1, zero_division=0)

print("\n================ CONFUSION MATRIX ================")
print("Rows = Actual, Columns = Predicted")
print(f"Classes: {CLASS_NAMES}")
print(cm)

tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

print("\n================ DETAILED METRICS ================")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n================ BINARY BREAKDOWN ================")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\n================ CLASSIFICATION REPORT ================")
print(classification_report(
    y_true[:len(y_pred)],
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))