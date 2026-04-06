import tensorflow as tf
import glob

f = glob.glob('data/tfrecords/train/*.tfrecord')[0]
raw_dataset = tf.data.TFRecordDataset(f)
example = next(iter(raw_dataset))
parsed_example = tf.train.Example.FromString(example.numpy())

for k, v in parsed_example.features.feature.items():
    if v.HasField('bytes_list'):
        print(f"{k}: bytes_list (len {len(v.bytes_list.value[0]) if v.bytes_list.value else 0})")
    elif v.HasField('float_list'):
        print(f"{k}: float_list (len {len(v.float_list.value)})")
    elif v.HasField('int64_list'):
        print(f"{k}: int64_list (len {len(v.int64_list.value)}, val {v.int64_list.value})")
