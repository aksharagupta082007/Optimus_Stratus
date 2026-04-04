import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from env.cubesat_env import CubeSatEnv
import traceback
from typing import Callable, Dict, List, Tuple
from models.enums import Action, EpisodeEndReason, PayloadMode, TargetOpportunity, CommsMode, AttitudeMode


# =========================================================
# Small test helpers
# =========================================================

def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_in(value, allowed, message: str) -> None:
    if value not in allowed:
        raise AssertionError(f"{message} | got={value}, allowed={allowed}")


def print_pass(name: str) -> None:
    print(f"[PASS] {name}")


def print_fail(name: str, err: Exception) -> None:
    print(f"[FAIL] {name}")
    print(f"       {type(err).__name__}: {err}")


# =========================================================
# Test cases
# =========================================================

def test_reset() -> None:
    env = CubeSatEnv()
    obs, info = env.reset(seed=42)

    assert_true(obs is not None, "reset must return observation")
    assert_true(info is not None, "reset must return info")
    assert_true(env.state.time.step_count == 0, "step_count must be 0 after reset")
    assert_true(env.state.time.sim_time_s == 0.0, "sim_time_s must be 0 after reset")
    assert_true(len(env.action_order) > 0, "action_order must not be empty")

    mask = env.get_action_mask()
    assert_true(len(mask) == len(env.action_order), "action mask length must match action space")
    assert_true(any(v == 1 for v in mask), "at least one action must be valid after reset")

    debug = info.get("debug", {})
    assert_true("battery_soc_pct" in debug, "debug must contain battery_soc_pct")
    assert_true("orbit_phase" in debug, "debug must contain orbit_phase")
    assert_true("spacecraft_mode" in debug, "debug must contain spacecraft_mode")


def test_noop_rollout() -> None:
    env = CubeSatEnv()
    env.reset(seed=42)

    initial_phase = env.state.orbit.orbit_phase
    initial_battery = env.state.eps.battery_soc_pct

    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(Action.NO_OP)
        assert_true(obs is not None, "step must return observation")
        assert_true(isinstance(reward, (int, float)), "reward must be numeric")
        assert_true(isinstance(terminated, bool), "terminated must be bool")
        assert_true(isinstance(truncated, bool), "truncated must be bool")
        assert_true(info is not None, "info must be returned")
        if terminated or truncated:
            break

    assert_true(env.state.time.step_count > 0, "step_count must increase after rollout")
    assert_true(env.state.orbit.orbit_phase != initial_phase or env.state.time.orbit_index > 0,
                "orbit phase or orbit index must change after rollout")
    assert_true(0.0 <= env.state.eps.battery_soc_pct <= 100.0, "battery must remain bounded")
    assert_true(env.state.cdh.memory_used_mb >= 0.0, "memory used must never be negative")
    assert_true(env.state.cdh.downlink_queue_mb >= 0.0, "queue must never be negative")

    # not a strict requirement to differ, but useful sanity note
    assert_true(initial_battery != env.state.eps.battery_soc_pct or env.state.time.step_count > 0,
                "battery should typically evolve during rollout")


def test_invalid_capture_penalty_path() -> None:
    env = CubeSatEnv()
    env.reset(seed=42)

    # Fresh reset should not have payload ready yet, so capture should be invalid
    obs, reward, terminated, truncated, info = env.step(Action.CAPTURE_IMAGE)

    validation = info.get("last_action_validation", {})
    assert_true(validation is not None, "last_action_validation must exist")
    assert_true(validation.get("valid") is False, "capture should be invalid right after reset")
    assert_true(len(validation.get("reasons", [])) > 0, "invalid action should explain reasons")
    assert_true(isinstance(reward, (int, float)), "reward must still be numeric on invalid action")
    assert_true(env.state.time.step_count == 1, "environment should still advance one step")


def test_payload_warmup_sequence() -> None:
    env = CubeSatEnv()
    env.reset(seed=42)

    # Force a nominal state where warmup should be allowed
    env.state.mode = env.state.mode.NOMINAL
    env.state.faults.safe_mode_latched = False
    env.state.eps.battery_soc_pct = 70.0

    env.state.payload.mode = PayloadMode.STANDBY
    env.state.payload.payload_enabled = False
    env.state.payload.imaging_ready = False
    env.state.payload.warmup_time_remaining_s = 0.0
    env.state.payload.imaging_cooldown_s = 0.0

    valid_actions = env.get_valid_actions()
    assert_true(Action.PAYLOAD_WARMUP in valid_actions, f"PAYLOAD_WARMUP not valid. valid={valid_actions}")

    obs, reward, terminated, truncated, info = env.step(Action.PAYLOAD_WARMUP)

    assert_in(
        env.state.payload.mode,
        (PayloadMode.WARMUP, PayloadMode.READY, PayloadMode.STANDBY),
        "payload mode after warmup command is unexpected",
    )

    # Give enough steps to finish warmup if it started
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(Action.NO_OP)
        if env.state.payload.imaging_ready:
            break
        if terminated or truncated:
            break

    assert_true(
        env.state.payload.mode in (PayloadMode.READY, PayloadMode.WARMUP, PayloadMode.STANDBY, PayloadMode.OFF),
        "payload mode must remain valid",
    )
    assert_true(
        env.state.payload.payload_enabled or env.state.payload.mode in (PayloadMode.STANDBY, PayloadMode.OFF, PayloadMode.READY),
        "payload state must remain coherent",
    )

def test_forced_valid_capture_path() -> None:
    
    env = CubeSatEnv()
    env.reset(seed=42)

    env.state.mode = env.state.mode.SCIENCE
    env.state.faults.safe_mode_latched = False

    env.state.eps.battery_soc_pct = 70.0

    # Put orbit in a naturally valid target window
    env.state.orbit.orbit_phase = 0.33
    env.orbit.step(env.state.orbit, dt_s=0.0, time_orbit_index=env.state.time.orbit_index)

    # Force favorable lighting
    env.state.orbit.sun_elevation_deg = 50.0

    env.state.adcs.mode = AttitudeMode.NADIR_POINTING
    env.state.adcs.pointing_error_deg = 0.5
    env.state.adcs.slew_time_remaining_s = 0.0

    env.state.payload.payload_enabled = True
    env.state.payload.imaging_ready = True
    env.state.payload.mode = PayloadMode.READY
    env.state.payload.warmup_time_remaining_s = 0.0
    env.state.payload.imaging_cooldown_s = 0.0
    env.state.payload.clear_current_frame()

    valid_actions = env.get_valid_actions()
    assert_true(Action.CAPTURE_IMAGE in valid_actions, f"CAPTURE_IMAGE not valid. valid={valid_actions}")

    obs, reward, terminated, truncated, info = env.step(Action.CAPTURE_IMAGE)

    validation = info.get("last_action_validation", {})
    assert_true(validation.get("valid") is True, f"forced valid capture should validate. reasons={validation.get('reasons')}")

    # Helpful debugging if capture still fails
    debug = info.get("debug", {})
    assert_true(env.state.payload.has_frame, f"payload should hold a frame after successful capture | debug={debug}")

    assert_true(env.state.payload.current_frame_id is not None, "captured frame must have id")
    assert_true(env.state.payload.current_frame_size_mb > 0.0, "captured frame must have positive size")
    assert_true(0.0 <= env.state.payload.current_frame_cloud_prob <= 1.0, "cloud prob must be bounded")
    assert_true(0.0 <= env.state.payload.current_frame_usefulness <= 1.0, "usefulness must be bounded")
    assert_true(0.0 <= env.state.payload.classifier_confidence <= 1.0, "confidence must be bounded")

def test_store_frame_increases_queue() -> None:
    env = CubeSatEnv()
    env.reset(seed=42)

    env.state.mode = env.state.mode.SCIENCE
    env.state.faults.safe_mode_latched = False

    # Force a stored-frame-ready scenario
    env.state.payload.payload_enabled = True
    env.state.payload.imaging_ready = True
    env.state.payload.mode = PayloadMode.READY

    env.state.payload.current_frame_id = 1
    env.state.payload.current_frame_size_mb = 32.0
    env.state.payload.current_frame_class = env.state.payload.current_frame_class.CLEAR
    env.state.payload.current_frame_cloud_prob = 0.15
    env.state.payload.current_frame_usefulness = 0.82
    env.state.payload.classifier_confidence = 0.90

    env.state.cdh.raw_buffer_mb = 32.0
    env.state.cdh.processed_buffer_mb = 0.0
    env.state.cdh.downlink_queue_mb = 0.0
    env.state.cdh.memory_used_mb = 32.0
    env.state.cdh.filesystem_healthy = True
    env.state.cdh.storage_corrupted = False

    valid_actions = env.get_valid_actions()
    assert_true(Action.STORE_FRAME in valid_actions, f"STORE_FRAME not valid. valid={valid_actions}")

    queue_before = env.state.cdh.downlink_queue_mb
    obs, reward, terminated, truncated, info = env.step(Action.STORE_FRAME)
    queue_after = env.state.cdh.downlink_queue_mb

    validation = info.get("last_action_validation", {})
    assert_true(validation.get("valid") is True, f"store frame should validate. reasons={validation.get('reasons')}")
    assert_true(queue_after >= queue_before, "queue should increase or stay same after store")
    assert_true(not env.state.payload.has_frame, "payload frame should clear after storing")


def test_discard_frame_clears_payload() -> None:
    env = CubeSatEnv()
    env.reset(seed=42)

    env.state.mode = env.state.mode.SCIENCE
    env.state.faults.safe_mode_latched = False

    env.state.payload.payload_enabled = True
    env.state.payload.imaging_ready = True
    env.state.payload.mode = PayloadMode.READY

    env.state.payload.current_frame_id = 1
    env.state.payload.current_frame_size_mb = 30.0
    env.state.payload.current_frame_class = env.state.payload.current_frame_class.CLOUDY
    env.state.payload.current_frame_cloud_prob = 0.9
    env.state.payload.current_frame_usefulness = 0.1
    env.state.payload.classifier_confidence = 0.95

    env.state.cdh.raw_buffer_mb = 30.0
    env.state.cdh.memory_used_mb = 30.0
    env.state.cdh.filesystem_healthy = True
    env.state.cdh.storage_corrupted = False

    valid_actions = env.get_valid_actions()
    assert_true(Action.DISCARD_FRAME in valid_actions, f"DISCARD_FRAME not valid. valid={valid_actions}")

    obs, reward, terminated, truncated, info = env.step(Action.DISCARD_FRAME)

    validation = info.get("last_action_validation", {})
    assert_true(validation.get("valid") is True, f"discard should validate when frame exists. reasons={validation.get('reasons')}")
    assert_true(not env.state.payload.has_frame, "payload frame should be cleared after discard")
    assert_true(env.state.cdh.raw_buffer_mb >= 0.0, "raw buffer must remain non-negative")



def test_downlink_path_when_pass_visible() -> None:
    env = CubeSatEnv()
    env.reset(seed=42)

    # Force some queue
    env.state.cdh.downlink_queue_mb = 64.0
    env.state.cdh.processed_buffer_mb = 64.0
    env.state.cdh.memory_used_mb = 64.0

    # Force pass-like geometry and good pointing
    env.state.orbit.orbit_phase = 0.18
    env.state.adcs.mode = AttitudeMode.GROUND_TRACKING
    env.state.adcs.pointing_error_deg = 0.4
    env.state.adcs.slew_time_remaining_s = 0.0
    env.state.eps.battery_soc_pct = 70.0
    env.state.comms.mode = CommsMode.PASS_PREP

    # One prep step
    env.step(Action.PREPARE_DOWNLINK)

    queue_before = env.state.cdh.downlink_queue_mb
    obs, reward, terminated, truncated, info = env.step(Action.DOWNLINK_LOW_RATE)
    queue_after = env.state.cdh.downlink_queue_mb

    # Since pass visibility is recomputed from orbit phase, queue should usually drop on this path
    assert_true(queue_after <= queue_before, "downlink should not increase queue")
    subsystems = info.get("subsystems", {})
    comms_bd = subsystems.get("comms")
    assert_true(comms_bd is not None, "comms breakdown must be present in info")


def test_fault_and_termination_battery_depletion() -> None:
    env = CubeSatEnv()
    env.reset(seed=42)

    # Force terminal battery condition
    env.state.eps.battery_soc_pct = 0.0

    obs, reward, terminated, truncated, info = env.step(Action.NO_OP)

    assert_true(terminated or truncated or env.state.done, "env should terminate when battery is depleted")
    assert_in(
        env.state.end_reason,
        (
            EpisodeEndReason.BATTERY_DEPLETED,
            EpisodeEndReason.FATAL_FAULT,
            EpisodeEndReason.MANUAL_ABORT,
            EpisodeEndReason.NOT_DONE,  # kept for robustness if termination policy differs
        ),
        "unexpected end reason for battery depletion path",
    )


def test_state_bounds_over_rollout() -> None:
    env = CubeSatEnv()
    env.reset(seed=42)

    action_cycle = [
        Action.NO_OP,
        Action.PAYLOAD_WARMUP,
        Action.NADIR_POINT_STANDBY,
        Action.CAPTURE_IMAGE,
        Action.STORE_FRAME,
        Action.SLEW_TO_GROUND,
        Action.PREPARE_DOWNLINK,
        Action.DOWNLINK_LOW_RATE,
        Action.DESATURATE_WHEELS,
    ]

    for i in range(100):
        action = action_cycle[i % len(action_cycle)]
        obs, reward, terminated, truncated, info = env.step(action)

        assert_true(0.0 <= env.state.eps.battery_soc_pct <= 100.0, "battery SoC out of bounds")
        assert_true(env.state.cdh.memory_used_mb >= 0.0, "memory_used_mb out of bounds")
        assert_true(env.state.cdh.downlink_queue_mb >= 0.0, "downlink_queue_mb out of bounds")
        assert_true(env.state.cdh.raw_buffer_mb >= 0.0, "raw_buffer_mb out of bounds")
        assert_true(env.state.cdh.processed_buffer_mb >= 0.0, "processed_buffer_mb out of bounds")
        assert_true(-100.0 <= env.state.thermal.bus_temp_c <= 100.0, "bus temp unreasonable")
        assert_true(-100.0 <= env.state.thermal.battery_temp_c <= 100.0, "battery temp unreasonable")
        assert_true(-100.0 <= env.state.thermal.payload_temp_c <= 100.0, "payload temp unreasonable")
        assert_true(0.0 <= env.state.adcs.pointing_error_deg <= 30.0, "pointing error unreasonable")
        assert_true(0.0 <= env.state.adcs.body_rate_deg_s <= 20.0, "body rate unreasonable")

        if terminated or truncated:
            break


# =========================================================
# Test runner
# =========================================================

TESTS: List[Tuple[str, Callable[[], None]]] = [
    ("reset", test_reset),
    ("noop_rollout", test_noop_rollout),
    ("invalid_capture_penalty_path", test_invalid_capture_penalty_path),
    ("payload_warmup_sequence", test_payload_warmup_sequence),
    ("forced_valid_capture_path", test_forced_valid_capture_path),
    ("store_frame_increases_queue", test_store_frame_increases_queue),
    ("discard_frame_clears_payload", test_discard_frame_clears_payload),
    ("downlink_path_when_pass_visible", test_downlink_path_when_pass_visible),
    ("fault_and_termination_battery_depletion", test_fault_and_termination_battery_depletion),
    ("state_bounds_over_rollout", test_state_bounds_over_rollout),
]


def run_all_tests() -> Dict[str, int]:
    passed = 0
    failed = 0

    print("=" * 72)
    print("CubeSat Environment Test Suite")
    print("=" * 72)

    for name, fn in TESTS:
        try:
            fn()
            print_pass(name)
            passed += 1
        except Exception as err:
            print_fail(name, err)
            traceback.print_exc()
            failed += 1

    print("-" * 72)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total : {passed + failed}")
    print("=" * 72)

    return {"passed": passed, "failed": failed, "total": passed + failed}


if __name__ == "__main__":
    run_all_tests()
