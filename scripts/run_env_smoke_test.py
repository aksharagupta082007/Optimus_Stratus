from __future__ import annotations

import os
import sys
from typing import List

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from env.cubesat_env import CubeSatEnv
from models.enums import Action


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_step_summary(step: int, reward: float, info: dict) -> None:
    debug = info.get("debug", {})
    notes = info.get("notes", [])
    faults = info.get("faults_active", [])

    print(
        f"Step {step:4d} | "
        f"Mode={debug.get('spacecraft_mode', 'NA'):>18} | "
        f"Action={debug.get('last_action', 'NA'):>20} | "
        f"Reward={reward:7.3f} | "
        f"Battery={debug.get('battery_soc_pct', 0.0):6.2f}% | "
        f"Queue={debug.get('downlink_queue_mb', 0.0):7.2f} MB | "
        f"Sun={debug.get('sunlight_state', 'NA'):>8} | "
        f"Target={debug.get('target_opportunity', 'NA'):>12} | "
        f"Pass={debug.get('ground_pass_state', 'NA'):>14}"
    )

    if notes:
        print("   Notes:", " | ".join(notes[:3]))

    if faults:
        fault_str = ", ".join(
            f"{f['fault_type']}({f['level']})" for f in faults[:4]
        )
        print(f"   Active faults: {fault_str}")


def choose_demo_action(step: int) -> Action:
    """
    Simple deterministic action cycle for smoke testing.
    This is not a good policy; it is just meant to exercise the env.
    """
    cycle: List[Action] = [
        Action.NO_OP,
        Action.PAYLOAD_WARMUP,
        Action.NADIR_POINT_STANDBY,
        Action.CAPTURE_IMAGE,
        Action.RUN_CLASSIFIER,
        Action.STORE_FRAME,
        Action.SLEW_TO_GROUND,
        Action.PREPARE_DOWNLINK,
        Action.DOWNLINK_LOW_RATE,
        Action.DESATURATE_WHEELS,
        Action.NO_OP,
    ]
    return cycle[step % len(cycle)]


def print_final_summary(env: CubeSatEnv, total_reward: float, steps_run: int, done: bool, info: dict) -> None:
    debug = info.get("debug", {})
    episode_stats = info.get("episode_stats", {})

    print_header("FINAL SMOKE TEST SUMMARY")
    print(f"Steps run              : {steps_run}")
    print(f"Done                   : {done}")
    print(f"End reason             : {env.state.end_reason.name}")
    print(f"Total reward           : {total_reward:.3f}")
    print(f"Final spacecraft mode  : {debug.get('spacecraft_mode', 'NA')}")
    print(f"Final battery          : {debug.get('battery_soc_pct', 0.0):.2f}%")
    print(f"Final memory used      : {debug.get('memory_used_mb', 0.0):.2f} MB")
    print(f"Final downlink queue   : {debug.get('downlink_queue_mb', 0.0):.2f} MB")
    print(f"Final payload mode     : {debug.get('payload_mode', 'NA')}")
    print(f"Final frame class      : {debug.get('frame_class', 'NA')}")
    print(f"Final pass state       : {debug.get('ground_pass_state', 'NA')}")
    print(f"Final link quality     : {debug.get('link_quality', 'NA')}")
    print(f"Fault count            : {episode_stats.get('fault_count', 0)}")
    print(f"Useful images captured : {episode_stats.get('useful_images_captured', 0)}")
    print(f"Useful images stored   : {episode_stats.get('useful_images_stored', 0)}")
    print(f"Useful images downlinked: {episode_stats.get('useful_images_downlinked', 0)}")
    print(f"Total downlinked MB    : {episode_stats.get('total_data_downlinked_mb', 0.0)}")


def main() -> None:
    print_header("CUBESAT ENVIRONMENT SMOKE TEST")

    env = CubeSatEnv()
    obs, info = env.reset(seed=42)

    print("Reset successful.")
    print("Initial valid actions:")
    print(" ", [a.name for a in env.get_valid_actions()])

    max_steps = 150
    total_reward = 0.0
    done = False

    for step in range(1, max_steps + 1):
        action = choose_demo_action(step)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step <= 10 or step % 10 == 0 or terminated or truncated:
            print_step_summary(step, reward, info)

        done = terminated or truncated
        if done:
            break

    print_final_summary(env, total_reward, step, done, info)


if __name__ == "__main__":
    main()