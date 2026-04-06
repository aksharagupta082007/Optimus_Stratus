import tensorflow as tf

# Mirrors PayloadConfig defaults — update if PayloadConfig changes
_MIN_PAYLOAD_TEMP_C = -5.0
_MAX_PAYLOAD_TEMP_C = 40.0
IMG_SIZE = 96  # optimal for STM32H743 memory budget

# ── PC training: parse TFRecord from dataset ──────────────────────────────
def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    width = tf.cast(parsed['width'], tf.int32)
    height = tf.cast(parsed['height'], tf.int32)
    
    image = tf.io.decode_raw(parsed['image'], tf.float16)
    image = tf.reshape(image, [height, width, 3])
    image = tf.cast(image, tf.float32)
    
    label = tf.io.decode_raw(parsed['label'], tf.uint8)
    label = tf.cast(label, tf.float32)
    
    # Convert segmentation mask to binary image label (1 if >20% cloud, else 0)
    img_label = tf.cast(tf.reduce_mean(label) > 0.2, tf.float32)
    
    return image, img_label

def preprocess_train(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # Values might already be reflectance, clip appropriately
    # Optionally normalize to stretch contrast
    img_max = tf.reduce_max(image)
    image = tf.cond(img_max > 0, lambda: image / img_max, lambda: image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

def preprocess_val(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    img_max = tf.reduce_max(image)
    image = tf.cond(img_max > 0, lambda: image / img_max, lambda: image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def build_dataset(tfrecord_paths, training=True, batch_size=32):
    ds = tf.data.TFRecordDataset(tfrecord_paths, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    preprocess_fn = preprocess_train if training else preprocess_val
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ── On-device (STM32 simulation): preprocess a raw numpy frame ────────────────
import numpy as np

def preprocess_frame(raw_frame_rgb: np.ndarray) -> np.ndarray:
    """
    Takes a (H, W, 3) uint8 numpy array from the camera,
    returns a (1, 96, 96, 3) float32 array ready for TFLite.
    On actual STM32 this logic is reimplemented in C via STM32CubeAI.
    """
    import cv2
    img = cv2.resize(raw_frame_rgb, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)
