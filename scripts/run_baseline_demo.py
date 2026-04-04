from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from env.cubesat_env import CubeSatEnv
from models.enums import Action
from rl.baseline_policy import BaselineDecision, BaselinePolicy, BaselinePolicyConfig


# =========================================================
# Demo configuration
# =========================================================

@dataclass(frozen=True)
class BaselineDemoConfig:
    """
    Configuration for running the rule-based baseline policy against the
    CubeSat environment in a readable, debugging-friendly way.
    """
    episodes: int = 3
    max_steps_per_episode: int = 250
    seed: int = 42

    # Printing / logging behavior
    print_every_step_first_n: int = 20
    print_every_n_steps_after: int = 10
    print_on_action_change: bool = True
    print_on_notes: bool = True
    print_on_events: bool = True

    # End-of-run summaries
    print_episode_summary: bool = True
    print_final_summary: bool = True


DEFAULT_BASELINE_DEMO_CONFIG = BaselineDemoConfig()


# =========================================================
# Generic helpers
# =========================================================

def hr(char: str = "=", width: int = 108) -> str:
    return char * width


def safe_get(d: Any, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def enum_or_value_name(value: Any) -> str:
    if hasattr(value, "name"):
        return value.name
    return str(value)


def format_faults(faults: Sequence[Any]) -> str:
    if not faults:
        return "None"

    out: List[str] = []
    for f in faults[:5]:
        if isinstance(f, dict):
            fault_type = safe_get(f, "fault_type", "UNKNOWN")
            level = safe_get(f, "level", "UNKNOWN")
            out.append(f"{fault_type}({level})")
        else:
            out.append(str(f))
    return ", ".join(out)


# =========================================================
# Debug extraction helpers
# =========================================================

def get_debug(info: dict) -> dict:
    return safe_get(info, "debug", {}) or {}


def debug_value(debug: dict, *keys: str, default=None):
    for key in keys:
        if key in debug:
            return debug[key]
    return default


def format_valid_actions(valid_actions: Sequence[Any]) -> str:
    names: List[str] = []
    for a in valid_actions:
        if isinstance(a, Action):
            names.append(a.name)
        else:
            names.append(str(a))
    return ", ".join(names)


# =========================================================
# Event detection helpers
# =========================================================

def detect_events(prev_info: dict, curr_info: dict) -> List[str]:
    events: List[str] = []

    prev_debug = get_debug(prev_info)
    curr_debug = get_debug(curr_info)

    prev_queue = as_float(
        debug_value(prev_debug, "downlink_queue_mb", "queue_mb", default=0.0)
    )
    curr_queue = as_float(
        debug_value(curr_debug, "downlink_queue_mb", "queue_mb", default=0.0)
    )

    prev_mem = as_float(
        debug_value(prev_debug, "memory_used_mb", "storage_used_mb", default=0.0)
    )
    curr_mem = as_float(
        debug_value(curr_debug, "memory_used_mb", "storage_used_mb", default=0.0)
    )

    prev_frame = enum_or_value_name(
        debug_value(prev_debug, "frame_class", default="NONE")
    )
    curr_frame = enum_or_value_name(
        debug_value(curr_debug, "frame_class", default="NONE")
    )

    prev_pass = enum_or_value_name(
        debug_value(prev_debug, "ground_pass_state", "pass_state", default="NONE")
    )
    curr_pass = enum_or_value_name(
        debug_value(curr_debug, "ground_pass_state", "pass_state", default="NONE")
    )

    prev_faults_raw = safe_get(prev_info, "faults_active", []) or []
    curr_faults_raw = safe_get(curr_info, "faults_active", []) or []

    prev_faults = set()
    curr_faults = set()

    for f in prev_faults_raw:
        if isinstance(f, dict):
            prev_faults.add(f"{safe_get(f, 'fault_type', 'UNKNOWN')}({safe_get(f, 'level', 'UNKNOWN')})")
        else:
            prev_faults.add(str(f))

    for f in curr_faults_raw:
        if isinstance(f, dict):
            curr_faults.add(f"{safe_get(f, 'fault_type', 'UNKNOWN')}({safe_get(f, 'level', 'UNKNOWN')})")
        else:
            curr_faults.add(str(f))

    if prev_frame == "NONE" and curr_frame != "NONE":
        events.append(f"Frame appeared: {curr_frame}")

    if prev_frame != "NONE" and curr_frame == "NONE":
        if curr_mem > prev_mem + 1e-9:
            events.append("Frame stored into memory")
        else:
            events.append("Frame cleared/discarded")

    if curr_queue > prev_queue + 1e-9:
        events.append(f"Downlink queue increased by {curr_queue - prev_queue:.2f} MB")
    elif curr_queue < prev_queue - 1e-9:
        events.append(f"Downlink queue decreased by {prev_queue - curr_queue:.2f} MB")

    if prev_pass == "NONE" and curr_pass != "NONE":
        events.append(f"Ground pass acquired: {curr_pass}")
    elif prev_pass != "NONE" and curr_pass == "NONE":
        events.append("Ground pass lost")

    new_faults = curr_faults - prev_faults
    cleared_faults = prev_faults - curr_faults

    if new_faults:
        events.append(f"New fault(s): {', '.join(sorted(new_faults))}")
    if cleared_faults:
        events.append(f"Cleared fault(s): {', '.join(sorted(cleared_faults))}")

    return events


# =========================================================
# Print policy
# =========================================================

def should_print_step(
    step_idx: int,
    action_changed: bool,
    events: List[str],
    notes: Sequence[str],
    config: BaselineDemoConfig,
) -> bool:
    if step_idx <= config.print_every_step_first_n:
        return True
    if step_idx % config.print_every_n_steps_after == 0:
        return True
    if config.print_on_action_change and action_changed:
        return True
    if config.print_on_events and len(events) > 0:
        return True
    if config.print_on_notes and len(notes) > 0:
        return True
    return False


# =========================================================
# Pretty printers
# =========================================================

def print_episode_header(ep: int, total_eps: int) -> None:
    print("\n" + hr("="))
    print(f"BASELINE POLICY DEMO | EPISODE {ep}/{total_eps}")
    print(hr("="))


def print_initial_state(info: dict) -> None:
    debug = get_debug(info)
    valid_actions = safe_get(info, "valid_actions", []) or []

    mode = enum_or_value_name(debug_value(debug, "spacecraft_mode", "mode", default="NA"))
    battery = as_float(debug_value(debug, "battery_soc_pct", "battery_pct", default=0.0))
    payload = enum_or_value_name(debug_value(debug, "payload_mode", default="NA"))
    pass_state = enum_or_value_name(debug_value(debug, "ground_pass_state", "pass_state", default="NA"))
    target = enum_or_value_name(debug_value(debug, "target_opportunity", default="NA"))

    print("Initial state:")
    print(
        f"  Mode={mode} | "
        f"Battery={battery:.2f}% | "
        f"Payload={payload} | "
        f"Pass={pass_state} | "
        f"Target={target}"
    )
    print(f"  Valid actions: {format_valid_actions(valid_actions)}")


def print_step_line(
    step_idx: int,
    action: Action,
    decision: BaselineDecision,
    reward: float,
    info: dict,
) -> None:
    debug = get_debug(info)

    mode = enum_or_value_name(debug_value(debug, "spacecraft_mode", "mode", default="NA"))
    battery = as_float(debug_value(debug, "battery_soc_pct", "battery_pct", default=0.0))
    queue_mb = as_float(debug_value(debug, "downlink_queue_mb", "queue_mb", default=0.0))
    memory_mb = as_float(debug_value(debug, "memory_used_mb", default=0.0))
    frame_class = enum_or_value_name(debug_value(debug, "frame_class", default="NA"))
    pass_state = enum_or_value_name(debug_value(debug, "ground_pass_state", "pass_state", default="NA"))
    link_quality = enum_or_value_name(debug_value(debug, "link_quality", default="NA"))
    payload_mode = enum_or_value_name(debug_value(debug, "payload_mode", default="NA"))
    target = enum_or_value_name(debug_value(debug, "target_opportunity", default="NA"))
    pointing = enum_or_value_name(debug_value(debug, "pointing_quality", default="NA"))

    print(
        f"Step {step_idx:4d} | "
        f"Action={action.name:>20} | "
        f"Priority={decision.priority:>18} | "
        f"Reward={reward:8.3f} | "
        f"Mode={mode:>18} | "
        f"Battery={battery:6.2f}% | "
        f"Queue={queue_mb:7.2f} MB | "
        f"Mem={memory_mb:7.2f} MB"
    )
    print(
        f"   Context: Payload={payload_mode} | Frame={frame_class} | "
        f"Target={target} | Pass={pass_state} | Link={link_quality} | Pointing={pointing}"
    )
    print(f"   Reason : {decision.reason}")

    notes = safe_get(info, "notes", []) or []
    if notes:
        print(f"   Notes  : {' | '.join(str(x) for x in notes[:5])}")

    faults = safe_get(info, "faults_active", []) or []
    if faults:
        print(f"   Faults : {format_faults(faults)}")


def print_events(events: List[str]) -> None:
    for event in events:
        print(f"   Event  : {event}")


def print_episode_summary(
    ep: int,
    total_reward: float,
    steps_run: int,
    done: bool,
    info: dict,
) -> None:
    debug = get_debug(info)
    stats = safe_get(info, "episode_stats", {}) or {}
    faults = safe_get(info, "faults_active", []) or []

    print("\n" + hr("-"))
    print(f"EPISODE {ep} SUMMARY")
    print(hr("-"))
    print(f"Steps run               : {steps_run}")
    print(f"Done                    : {done}")
    print(f"End reason              : {enum_or_value_name(safe_get(info, 'end_reason', 'NA'))}")
    print(f"Total reward            : {total_reward:.3f}")
    print(f"Final spacecraft mode   : {enum_or_value_name(debug_value(debug, 'spacecraft_mode', 'mode', default='NA'))}")
    print(f"Final battery           : {as_float(debug_value(debug, 'battery_soc_pct', 'battery_pct', default=0.0)):.2f}%")
    print(f"Final memory used       : {as_float(debug_value(debug, 'memory_used_mb', default=0.0)):.2f} MB")
    print(f"Final downlink queue    : {as_float(debug_value(debug, 'downlink_queue_mb', 'queue_mb', default=0.0)):.2f} MB")
    print(f"Final payload mode      : {enum_or_value_name(debug_value(debug, 'payload_mode', default='NA'))}")
    print(f"Final frame class       : {enum_or_value_name(debug_value(debug, 'frame_class', default='NA'))}")
    print(f"Final pass state        : {enum_or_value_name(debug_value(debug, 'ground_pass_state', 'pass_state', default='NA'))}")
    print(f"Final link quality      : {enum_or_value_name(debug_value(debug, 'link_quality', default='NA'))}")
    print(f"Active faults           : {format_faults(faults)}")

    # Useful / mission stats
    print(f"Useful images captured  : {as_int(safe_get(stats, 'useful_images_captured', 0))}")
    print(f"Useful images stored    : {as_int(safe_get(stats, 'useful_images_stored', 0))}")
    print(f"Useful images downlinked: {as_int(safe_get(stats, 'useful_images_downlinked', 0))}")
    print(f"Total downlinked MB     : {as_float(safe_get(stats, 'total_data_downlinked_mb', 0.0)):.2f}")

    # Extra debug counters if they exist
    print(f"Total images captured   : {as_int(safe_get(stats, 'total_images_captured', 0))}")
    print(f"Total images stored     : {as_int(safe_get(stats, 'total_images_stored', 0))}")
    print(f"Total images downlinked : {as_int(safe_get(stats, 'total_images_downlinked', 0))}")
    print(f"Cloudy images captured  : {as_int(safe_get(stats, 'cloudy_images_captured', 0))}")
    print(f"Cloudy images stored    : {as_int(safe_get(stats, 'cloudy_images_stored', 0))}")
    print(f"Invalid actions         : {as_int(safe_get(stats, 'invalid_actions', 0))}")
    print(f"Capture attempts        : {as_int(safe_get(stats, 'capture_attempts', 0))}")
    print(f"Store attempts          : {as_int(safe_get(stats, 'store_attempts', 0))}")
    print(f"Downlink attempts       : {as_int(safe_get(stats, 'downlink_attempts', 0))}")


def print_final_multi_episode_summary(results: List[Dict[str, Any]]) -> None:
    print("\n" + hr("="))
    print("FINAL BASELINE DEMO SUMMARY")
    print(hr("="))

    if not results:
        print("No results available.")
        return

    avg_reward = sum(as_float(r["total_reward"]) for r in results) / len(results)
    avg_steps = sum(as_int(r["steps_run"]) for r in results) / len(results)
    avg_downlinked = sum(as_float(r["downlinked_mb"]) for r in results) / len(results)
    avg_useful_captured = sum(as_int(r["useful_captured"]) for r in results) / len(results)
    avg_useful_stored = sum(as_int(r["useful_stored"]) for r in results) / len(results)
    avg_useful_downlinked = sum(as_int(r["useful_downlinked"]) for r in results) / len(results)
    avg_invalid = sum(as_int(r["invalid_actions"]) for r in results) / len(results)

    for r in results:
        print(
            f"Episode {as_int(r['episode']):2d} | "
            f"Reward={as_float(r['total_reward']):9.3f} | "
            f"Steps={as_int(r['steps_run']):4d} | "
            f"Downlinked={as_float(r['downlinked_mb']):7.2f} MB | "
            f"UsefulCaptured={as_int(r['useful_captured']):3d} | "
            f"UsefulStored={as_int(r['useful_stored']):3d} | "
            f"UsefulDownlinked={as_int(r['useful_downlinked']):3d} | "
            f"Invalid={as_int(r['invalid_actions']):3d}"
        )

    print(hr("-"))
    print(f"Average reward            : {avg_reward:.3f}")
    print(f"Average steps             : {avg_steps:.2f}")
    print(f"Average total downlinked  : {avg_downlinked:.2f} MB")
    print(f"Average useful captured   : {avg_useful_captured:.2f}")
    print(f"Average useful stored     : {avg_useful_stored:.2f}")
    print(f"Average useful downlinked : {avg_useful_downlinked:.2f}")
    print(f"Average invalid actions   : {avg_invalid:.2f}")


# =========================================================
# Main runner
# =========================================================

def run_baseline_episode(
    episode_idx: int,
    total_episodes: int,
    env: CubeSatEnv,
    policy: BaselinePolicy,
    config: BaselineDemoConfig,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=config.seed + episode_idx)

    print_episode_header(episode_idx, total_episodes)
    print_initial_state(info)

    prev_info = info
    prev_action: Optional[Action] = None
    total_reward = 0.0
    done = False
    step_idx = 0

    for step_idx in range(1, config.max_steps_per_episode + 1):
        valid_actions = env.get_valid_actions()
        decision = policy.decide(env.state, valid_actions)
        action = decision.action

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        events = detect_events(prev_info, info)
        notes = safe_get(info, "notes", []) or []
        action_changed = prev_action != action

        if should_print_step(
            step_idx=step_idx,
            action_changed=action_changed,
            events=events,
            notes=notes,
            config=config,
        ):
            print_step_line(
                step_idx=step_idx,
                action=action,
                decision=decision,
                reward=reward,
                info=info,
            )
            if events:
                print_events(events)

        prev_info = info
        prev_action = action

        if done:
            break

    if config.print_episode_summary:
        print_episode_summary(
            ep=episode_idx,
            total_reward=total_reward,
            steps_run=step_idx,
            done=done,
            info=info,
        )

    stats = safe_get(info, "episode_stats", {}) or {}
    return {
        "episode": episode_idx,
        "total_reward": total_reward,
        "steps_run": step_idx,
        "downlinked_mb": as_float(safe_get(stats, "total_data_downlinked_mb", 0.0)),
        "useful_captured": as_int(safe_get(stats, "useful_images_captured", 0)),
        "useful_stored": as_int(safe_get(stats, "useful_images_stored", 0)),
        "useful_downlinked": as_int(safe_get(stats, "useful_images_downlinked", 0)),
        "invalid_actions": as_int(safe_get(stats, "invalid_actions", 0)),
    }

def print_step_line(
    step_idx: int,
    action: Action,
    decision: BaselineDecision,
    reward: float,
    info: dict,
) -> None:
    debug = get_debug(info)

    mode = enum_or_value_name(debug_value(debug, "spacecraft_mode", "mode", default="NA"))
    battery = as_float(debug_value(debug, "battery_soc_pct", "battery_pct", default=0.0))
    queue_mb = as_float(debug_value(debug, "downlink_queue_mb", "queue_mb", default=0.0))
    memory_mb = as_float(debug_value(debug, "memory_used_mb", default=0.0))
    frame_class = enum_or_value_name(debug_value(debug, "frame_class", default="NA"))
    pass_state = enum_or_value_name(debug_value(debug, "ground_pass_state", "pass_state", default="NA"))
    link_quality = enum_or_value_name(debug_value(debug, "link_quality", default="NA"))
    payload_mode = enum_or_value_name(debug_value(debug, "payload_mode", default="NA"))
    target = enum_or_value_name(debug_value(debug, "target_opportunity", default="NA"))
    pointing = enum_or_value_name(debug_value(debug, "pointing_quality", default="NA"))

    print(
        f"Step {step_idx:4d} | "
        f"Action={action.name:>20} | "
        f"Priority={decision.priority:>18} | "
        f"Reward={reward:8.3f} | "
        f"Mode={mode:>18} | "
        f"Battery={battery:6.2f}% | "
        f"Queue={queue_mb:7.2f} MB | "
        f"Mem={memory_mb:7.2f} MB"
    )
    print(
        f"   Context: Payload={payload_mode} | Frame={frame_class} | "
        f"Target={target} | Pass={pass_state} | Link={link_quality} | Pointing={pointing}"
    )
    print(f"   Reason : {decision.reason}")

    notes = safe_get(info, "notes", []) or []
    if notes:
        print(f"   Notes  : {' | '.join(str(x) for x in notes[:5])}")

    faults = safe_get(info, "faults_active", []) or []
    if faults:
        print(f"   Faults : {format_faults(faults)}")

    # NEW: print valid actions when they matter for debugging
    valid_actions = safe_get(info, "valid_actions", []) or []
    should_show_valid = (
        bool(faults)
        or target in {"VALID", "HIGH_VALUE"}
        or pass_state not in {"NONE", "NA"}
        or mode in {"SAFE", "FAULT_RECOVERY"}
    )
    if should_show_valid and valid_actions:
        print(f"   Valid  : {format_valid_actions(valid_actions)}")


def main() -> None:
    demo_config = DEFAULT_BASELINE_DEMO_CONFIG
    policy_config = BaselinePolicyConfig()

    env = CubeSatEnv()
    policy = BaselinePolicy(config=policy_config)

    results: List[Dict[str, Any]] = []

    print(hr("="))
    print("RUNNING BASELINE POLICY DEMO")
    print(hr("="))
    print(
        f"Episodes={demo_config.episodes} | "
        f"MaxSteps/Episode={demo_config.max_steps_per_episode} | "
        f"Seed={demo_config.seed}"
    )

    for ep in range(1, demo_config.episodes + 1):
        result = run_baseline_episode(
            episode_idx=ep,
            total_episodes=demo_config.episodes,
            env=env,
            policy=policy,
            config=demo_config,
        )
        results.append(result)

    if demo_config.print_final_summary:
        print_final_multi_episode_summary(results)


if __name__ == "__main__":
    main()