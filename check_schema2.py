import tensorflow as tf
import glob
import numpy as np

f = glob.glob('data/tfrecords/train/*.tfrecord')[0]
raw_dataset = tf.data.TFRecordDataset(f)
example = next(iter(raw_dataset))
parsed_example = tf.train.Example.FromString(example.numpy())

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'channels': tf.io.FixedLenFeature([], tf.int64),
}
parsed = tf.io.parse_single_example(example, feature_description)
image = tf.io.decode_raw(parsed['image'], tf.float16)
image = tf.reshape(image, [64, 64, 3])
label = tf.io.decode_raw(parsed['label'], tf.uint8)
label = tf.reshape(label, [64, 64])

print("Image shape:", image.shape)
print("Label shape:", label.shape)
print("Image min/max:", np.min(image.numpy()), np.max(image.numpy()))
print("Label min/max:", np.min(label.numpy()), np.max(label.numpy()))
