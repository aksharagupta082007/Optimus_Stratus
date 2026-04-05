import tensorflow as tf
import glob, os

IMG_SIZE   = 96
BATCH_SIZE = 32
EPOCHS_HEAD   = 10
EPOCHS_FINETUNE = 15

def build_model():
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        minimalistic=True   # ReLU6 only — no hard-swish, MCU-safe
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

def train(tfrecord_dir: str, output_path: str = 'classifier/model.tflite'):
    from classifier.preprocess import build_dataset

    train_files = glob.glob(f'{tfrecord_dir}/train/*.tfrecord')
    val_files   = glob.glob(f'{tfrecord_dir}/val/*.tfrecord')

    train_ds = build_dataset(train_files, training=True,  batch_size=BATCH_SIZE)
    val_ds   = build_dataset(val_files,   training=False, batch_size=BATCH_SIZE)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint('models/best_classifier.keras',
                                           save_best_only=True)
    ]

    # Phase 1: head only
    print("Phase 1: training head...")
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_HEAD, callbacks=callbacks)

    # Phase 2: unfreeze last 20 layers of MobileNetV3 base
    print("Phase 2: fine-tuning...")
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_FINETUNE, callbacks=callbacks)

    # Convert to TFLite int8
    _convert_to_tflite(model, val_ds, output_path)
    print(f"Saved TFLite model → {output_path}")
    return model

def _convert_to_tflite(model, rep_dataset, output_path):
    def representative_dataset():
        for img_batch, _ in rep_dataset.take(100):
            for img in img_batch:
                yield [tf.expand_dims(img, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model size: {len(tflite_model)/1024:.1f} KB")