"""Integration smoke test: TFRecord -> TFLite classifier -> FrameClass -> Agent decision."""
import sys, os
sys.path.insert(0, '.')
import numpy as np
import tensorflow as tf

# ── 1. Test classifier module in isolation ───────────────────────────────
print("=" * 60)
print("1. Classifier module test (parse -> infer -> postprocess)")
print("=" * 60)

TFR_DIR = os.path.join('data', 'tfrecords', 'test')
tfrecord_files = [
    os.path.join(TFR_DIR, f)
    for f in os.listdir(TFR_DIR)
    if f.endswith('.tfrecord')
]
print("  TFRecords found:", len(tfrecord_files))

# Read first available TFRecord
from classifier.preprocess   import parse_tfrecord_numpy, prepare_input
from classifier.infer_tflite import CloudClassifier
from classifier.postprocess  import postprocess

first_raw = None
for raw in tf.data.TFRecordDataset([tfrecord_files[0]]).take(1):
    first_raw = raw.numpy()

img, label = parse_tfrecord_numpy(first_raw)
print("  Parsed image shape:", img.shape if img is not None else None, " label:", label)

clf    = CloudClassifier()
inp    = prepare_input(img)
pred   = clf.predict(inp)
result = postprocess(pred)
print("  Cloud prob:", round(pred['current_frame_cloud_prob'], 4))
print("  Frame class:", result['frame_class'].name)
print("  Confidence:", round(result['classifier_confidence'], 4))
print("  Latency (s):", round(pred['classifier_last_latency_s'], 4))

# ── 2. Test environment with injected TFRecord ───────────────────────────
print()
print("=" * 60)
print("2. Environment integration test (inject TFRecord -> RUN_CLASSIFIER)")
print("=" * 60)

from env.cubesat_env import CubeSatEnv, LocalRLConfig
from models.enums import (
    Action, TargetOpportunity, PayloadMode, SunlightState,
    SpacecraftMode, AttitudeMode, FrameClass
)

env = CubeSatEnv(
    rl_config=LocalRLConfig(observation_type='vector'),
    tfrecord_dir=TFR_DIR,
)
obs, _ = env.reset(seed=42)
print("  Env reset OK. Obs shape:", np.array(obs).shape)
print("  TFRecord pool:", len(env._tfrecord_files), "files")

# Directly inject a frame into payload state (simulate successful CAPTURE_IMAGE)
from models.enums import FrameClass
env.state.mode = SpacecraftMode.SCIENCE
env.state.eps.battery_soc_pct = 85.0
env.state.payload.mode = PayloadMode.READY
env.state.payload.payload_enabled = True
env.state.payload.imaging_ready = True
env.state.payload.current_frame_id = 1          # mark a frame as held
env.state.payload.current_frame_size_mb = 32.0
env.state.payload.current_frame_class = FrameClass.NONE  # unclassified
# Inject raw TFRecord bytes so RUN_CLASSIFIER picks them up
env.state.payload.current_frame_tfrecord_bytes = first_raw

# Run classifier
obs, reward, term, trunc, info = env.step(Action.RUN_CLASSIFIER)
fc = info['debug'].get('frame_class', '?')
print("  RUN_CLASSIFIER -> frame_class:", fc, "  reward:", round(reward, 3))

# Agent decision
if str(fc) in ('CLEAR', 'HIGH_VALUE_CLEAR'):
    obs, reward, _, _, _ = env.step(Action.STORE_FRAME)
    print("  STORE_FRAME -> queued for downlink  reward:", round(reward, 3))
else:
    obs, reward, _, _, _ = env.step(Action.DISCARD_FRAME)
    print("  DISCARD_FRAME -> memory freed  reward:", round(reward, 3))

print()
print("[PASS] All integration checks complete.")
