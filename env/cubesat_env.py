from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.comms_config import CommsConfig
from config.mission_config import MissionConfig
from config.power_config import PowerConfig
from config.rl_config import RLConfig
from config.thermal_config import ThermalConfig

from env.observation_builder import (
    DEFAULT_OBSERVATION_CONFIG,
    ObservationBuilder,
    ObservationConfig,
)
from env.reward import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    RewardEngine,
    TransitionContext,
    infer_transition_context,
)
from env.termination import (
    DEFAULT_TERMINATION_CONFIG,
    TerminationConfig,
    TerminationEngine,
    TerminationTracker,
    apply_termination_result,
)

from models.action_space import (
    DEFAULT_ACTION_ORDER,
    ActionValidationResult,
    apply_recommended_mode_transitions,
    get_action_spec,
    get_valid_actions,
    index_to_action,
    validate_action,
)
from models.enums import (
    Action,
    AttitudeMode,
    CommsMode,
    DataHandlingMode,
    EpisodeEndReason,
    FaultType,
    PayloadMode,
    SpacecraftMode,
)
from models.state_models import (
    SpacecraftState,
    StepInfo,
    create_default_spacecraft_state,
)

from subsystems.adcs import ADCSConfig, ADCSSubsystem
from subsystems.cdh import CDHConfig, CDHSubsystem
from subsystems.comms import CommsSubsystem
from subsystems.faults import FaultConfig, FaultSubsystem
from subsystems.orbit import OrbitConfig, OrbitPropagator
from subsystems.payload import PayloadConfig, PayloadSubsystem
from subsystems.power import PowerSubsystem
from subsystems.thermal import ThermalSubsystem


# =========================================================
# Lightweight local configs (used if project config files
# do not yet exist or are placeholders)
# =========================================================

@dataclass(frozen=True)
class LocalMissionConfig:
    """
    Minimal mission-level config used directly by the environment.
    """
    dt_s: float = 5.0
    max_episode_steps: int = 1080           # ~1 orbit at 90 min and dt=5 s
    max_episode_orbits: Optional[int] = None
    initial_battery_soc_pct: float = 70.0
    initial_orbit_phase: float = 0.02
    initial_bus_temp_c: float = 18.0
    initial_battery_temp_c: float = 12.0
    initial_payload_temp_c: float = 14.0
    initial_radio_temp_c: float = 16.0
    initial_pointing_error_deg: float = 8.0
    initial_body_rate_deg_s: float = 1.0
    initial_wheel_momentum_nms: float = 0.03
    initial_mode: SpacecraftMode = SpacecraftMode.BOOT
    randomize_initial_phase: bool = False


@dataclass(frozen=True)
class LocalRLConfig:
    """
    Minimal RL config used by the environment.
    """
    invalid_action_uses_soft_penalty: bool = True
    invalid_action_fallback_to_noop: bool = True
    include_action_mask_in_obs: bool = True
    observation_type: str = "bundle"  # "bundle", "tabular", "vector"


# =========================================================
# Environment
# =========================================================

class CubeSatEnv(gym.Env):
    """
    Integrated operational CubeSat environment.

    This environment ties together:
    - orbit progression
    - EPS/power
    - thermal
    - ADCS
    - payload
    - CDH/storage
    - communications
    - FDIR/faults
    - reward
    - termination
    - observation building

    Design goals:
    - realistic operational flow
    - deterministic structure + stochastic scene generation
    - easy to test and debug
    - directly usable for baseline policy, Q-learning, and DQN

    Step order:
    1. validate / accept action
    2. apply recommended mode transitions
    3. advance orbit
    4. advance ADCS
    5. advance payload
    6. ingest captured frame to CDH raw buffer if needed
    7. advance CDH
    8. advance comms
    9. advance power
    10. advance thermal
    11. advance faults
    12. compute reward
    13. compute termination
    14. build observation
    """

    metadata: Dict[str, Any] = {
        "name": "EarthObservationCubeSatSSOEnv",
        "render_modes": [],
    }

    def __init__(
        self,
        mission_config: Optional[Any] = None,
        rl_config: Optional[Any] = None,
        orbit_config: Optional[OrbitConfig] = None,
        power_config: Optional[PowerConfig] = None,
        thermal_config: Optional[ThermalConfig] = None,
        adcs_config: Optional[ADCSConfig] = None,
        comms_config: Optional[CommsConfig] = None,
        cdh_config: Optional[CDHConfig] = None,
        payload_config: Optional[PayloadConfig] = None,
        fault_config: Optional[FaultConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        termination_config: Optional[TerminationConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
    ) -> None:
        # Use project configs if they are passed; otherwise use local defaults.
        self.mission_config = mission_config or LocalMissionConfig()
        self.rl_config = rl_config or LocalRLConfig()

        # Patch termination defaults from mission config if needed
        if termination_config is None:
            termination_config = DEFAULT_TERMINATION_CONFIG
            termination_config = TerminationConfig(
                max_steps=getattr(self.mission_config, "max_episode_steps", 1080),
                max_orbits=getattr(self.mission_config, "max_episode_orbits", None),
                terminate_on_battery_depleted=termination_config.terminate_on_battery_depleted,
                battery_depleted_threshold_pct=termination_config.battery_depleted_threshold_pct,
                terminate_on_brownout_risk=termination_config.terminate_on_brownout_risk,
                terminate_on_sustained_critical_battery=termination_config.terminate_on_sustained_critical_battery,
                critical_battery_threshold_pct=termination_config.critical_battery_threshold_pct,
                critical_battery_max_consecutive_steps=termination_config.critical_battery_max_consecutive_steps,
                terminate_on_thermal_violation=termination_config.terminate_on_thermal_violation,
                terminate_on_sustained_battery_thermal_violation=termination_config.terminate_on_sustained_battery_thermal_violation,
                battery_temp_too_cold_c=termination_config.battery_temp_too_cold_c,
                battery_temp_too_hot_c=termination_config.battery_temp_too_hot_c,
                battery_thermal_max_consecutive_steps=termination_config.battery_thermal_max_consecutive_steps,
                terminate_on_sustained_payload_thermal_violation=termination_config.terminate_on_sustained_payload_thermal_violation,
                payload_temp_too_cold_c=termination_config.payload_temp_too_cold_c,
                payload_temp_too_hot_c=termination_config.payload_temp_too_hot_c,
                payload_thermal_max_consecutive_steps=termination_config.payload_thermal_max_consecutive_steps,
                terminate_on_sustained_bus_thermal_violation=termination_config.terminate_on_sustained_bus_thermal_violation,
                bus_temp_too_cold_c=termination_config.bus_temp_too_cold_c,
                bus_temp_too_hot_c=termination_config.bus_temp_too_hot_c,
                bus_thermal_max_consecutive_steps=termination_config.bus_thermal_max_consecutive_steps,
                terminate_on_fatal_fault=termination_config.terminate_on_fatal_fault,
                terminate_on_any_critical_fault=termination_config.terminate_on_any_critical_fault,
                terminate_on_specific_faults=termination_config.terminate_on_specific_faults,
                terminate_on_storage_failure=termination_config.terminate_on_storage_failure,
                terminate_on_memory_overflow=termination_config.terminate_on_memory_overflow,
                memory_overflow_threshold_frac=termination_config.memory_overflow_threshold_frac,
                terminate_on_filesystem_corruption=termination_config.terminate_on_filesystem_corruption,
                terminate_if_not_alive_flag_false=termination_config.terminate_if_not_alive_flag_false,
                terminate_if_done_flag_true=termination_config.terminate_if_done_flag_true,
                terminate_on_manual_abort_flag=termination_config.terminate_on_manual_abort_flag,
            )

        # Subsystems
        self.orbit = OrbitPropagator(config=orbit_config or OrbitConfig())
        self.power = PowerSubsystem(config=power_config or PowerConfig())
        self.thermal = ThermalSubsystem(config=thermal_config or ThermalConfig())
        self.adcs = ADCSSubsystem(config=adcs_config or ADCSConfig())
        self.comms = CommsSubsystem(config=comms_config or CommsConfig())
        self.cdh = CDHSubsystem(config=cdh_config or CDHConfig())
        self.payload = PayloadSubsystem(config=payload_config or PayloadConfig())
        self.faults = FaultSubsystem(config=fault_config or FaultConfig())

        # Support systems
        obs_cfg = observation_config or DEFAULT_OBSERVATION_CONFIG
        if getattr(self.rl_config, "include_action_mask_in_obs", True):
            obs_cfg = ObservationConfig(
                include_action_mask=True,
                include_debug_fields=obs_cfg.include_debug_fields,
                queue_norm_max_mb=obs_cfg.queue_norm_max_mb,
                battery_soc_thresholds_pct=obs_cfg.battery_soc_thresholds_pct,
                battery_temp_thresholds_c=obs_cfg.battery_temp_thresholds_c,
                payload_temp_thresholds_c=obs_cfg.payload_temp_thresholds_c,
                pointing_error_thresholds_deg=obs_cfg.pointing_error_thresholds_deg,
                sun_elevation_thresholds_deg=obs_cfg.sun_elevation_thresholds_deg,
                gs_elevation_thresholds_deg=obs_cfg.gs_elevation_thresholds_deg,
                wheel_momentum_util_thresholds=obs_cfg.wheel_momentum_util_thresholds,
                classifier_conf_thresholds=obs_cfg.classifier_conf_thresholds,
                usefulness_thresholds=obs_cfg.usefulness_thresholds,
                use_one_hot_for_categoricals=obs_cfg.use_one_hot_for_categoricals,
            )

        self.observation_builder = ObservationBuilder(config=obs_cfg)
        self.reward_engine = RewardEngine(config=reward_config or DEFAULT_REWARD_CONFIG)
        self.termination_engine = TerminationEngine(config=termination_config)

        # Runtime state
        self.state: SpacecraftState = create_default_spacecraft_state()
        self.termination_tracker = TerminationTracker()

        self.last_step_info: Optional[StepInfo] = None
        self.last_action_validation: Optional[ActionValidationResult] = None

        self.action_order: List[Action] = list(DEFAULT_ACTION_ORDER)
        self._manual_abort_requested: bool = False

        # --- Gymnasium Spaces ---
        self.action_space = spaces.Discrete(len(self.action_order))
        
        # We parse the output shape based on the configured observation type
        dummy_obs = self._format_observation()
        if isinstance(dummy_obs, (list, tuple)):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(dummy_obs),), dtype=np.float32
            )
        elif hasattr(dummy_obs, "shape"): # numpy arrays
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32
            )
        else:
            # If the observation is a complex bundle (dictionary), we use a placeholder space here.
            # (In a production Dict space, you would map each dict key to its own Box/Discrete space).
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    # -----------------------------------------------------
    # Gym-like API
    # -----------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset environment to a fresh episode.
        """
        options = options or {}

        # Rebuild a clean state
        self.state = create_default_spacecraft_state()
        self.termination_tracker.reset()
        self.last_step_info = None
        self.last_action_validation = None
        self._manual_abort_requested = False

        # Seed subsystem RNGs if requested
        if seed is not None:
            self._reset_rngs(seed)

        # Basic state initialization
        self._initialize_state(options)

        # Build first observation
        observation = self._format_observation()
        info = self._build_info_dict(done=False, reward=0.0, extra_notes=["reset completed"])
        return observation, info

    def step(
        self,
        action: int | Action,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.state.done:
            # Gym-compatible behavior: stepping a done env is an error in spirit,
            # but we keep it safe and explicit.
            observation = self._format_observation()
            info = self._build_info_dict(
                done=True,
                reward=0.0,
                extra_notes=["step called on terminated environment"],
            )
            return observation, 0.0, True, False, info

        resolved_action = self._resolve_action(action)
        self.state.begin_step(resolved_action)

        prev_state = copy.deepcopy(self.state)

        # Validate action
        validation = validate_action(self.state, resolved_action)
        self.last_action_validation = validation

        effective_action = resolved_action
        invalid_action = not validation.valid
        if invalid_action and getattr(self.rl_config, "invalid_action_fallback_to_noop", True):
            effective_action = Action.NO_OP

        # Action profile / intent
        action_spec = get_action_spec(effective_action)
        action_profile = action_spec.profile()

        # Apply recommended mode transitions before subsystem updates
        apply_recommended_mode_transitions(self.state, effective_action)

        # Promote some top-level mode semantics
        self._pre_step_mode_sanity(effective_action)

        # Advance time + orbit first
        self._advance_time_and_orbit()

        # ADCS
        adcs_breakdown = self.adcs.step(
            state=self.state,
            dt_s=self.state.time.dt_s,
            action=effective_action,
            action_profile=action_profile,
        )

        # Payload
        payload_breakdown = self.payload.step(
            state=self.state,
            dt_s=self.state.time.dt_s,
            action=effective_action,
            action_profile=action_profile,
        )

        # If a frame was generated, ingest raw frame into CDH
        if payload_breakdown.frame_generated and self.state.payload.has_frame:
            self.cdh.ingest_captured_frame(
                self.state,
                frame_size_mb=self.state.payload.current_frame_size_mb,
            )

        # CDH
        cdh_breakdown = self.cdh.step(
            state=self.state,
            dt_s=self.state.time.dt_s,
            action=effective_action,
            action_profile=action_profile,
        )

        # Comms
        comms_breakdown = self.comms.step(
            state=self.state,
            dt_s=self.state.time.dt_s,
            action=effective_action,
            action_profile=action_profile,
        )

        # Power
        power_breakdown = self.power.step(
            state=self.state,
            dt_s=self.state.time.dt_s,
            action=effective_action,
            action_profile=action_profile,
        )

        # Thermal
        thermal_breakdown = self.thermal.step(
            state=self.state,
            dt_s=self.state.time.dt_s,
            action=effective_action,
            action_profile=action_profile,
        )

        # Faults last among physical subsystems
        fault_breakdown = self.faults.step(
            state=self.state,
            dt_s=self.state.time.dt_s,
            action=effective_action,
        )

        # Boot/Nominal settling
        self._post_step_mode_sanity()

        # Termination
        termination_result = self.termination_engine.evaluate(
            state=self.state,
            tracker=self.termination_tracker,
            manual_abort=self._manual_abort_requested,
        )
        apply_termination_result(self.state, termination_result)

        # Reward
        transition_ctx = infer_transition_context(
            prev_state=prev_state,
            next_state=self.state,
            action=resolved_action,
            action_valid=validation.valid,
        )
        transition_ctx.end_reason = self.state.end_reason
        transition_ctx.became_done_this_step = self.state.done and not prev_state.done

        reward_breakdown = self.reward_engine.compute_reward(
            prev_state=prev_state,
            next_state=self.state,
            ctx=transition_ctx,
        )

        reward = reward_breakdown.total
        self.state.step_reward = reward_breakdown
        self.state.end_step(reward)

        # Step info
        step_info = StepInfo(
            action_taken=resolved_action,
            reward_breakdown=reward_breakdown,
            done=self.state.done,
            end_reason=self.state.end_reason,
            notes=[],
        )

        if invalid_action:
            step_info.add_note("invalid action received")
            for reason in validation.reasons:
                step_info.add_note(reason)

        if termination_result.reasons:
            for msg in termination_result.reasons:
                step_info.add_note(msg)

        self.last_step_info = step_info

        # Output
        observation = self._format_observation()
        terminated = self.state.done and self.state.end_reason != EpisodeEndReason.MAX_STEPS
        truncated = self.state.done and self.state.end_reason == EpisodeEndReason.MAX_STEPS
        info = self._build_info_dict(
            done=self.state.done,
            reward=reward,
            extra_notes=step_info.notes,
            subsystem_debug={
                "adcs": adcs_breakdown,
                "payload": payload_breakdown,
                "cdh": cdh_breakdown,
                "comms": comms_breakdown,
                "power": power_breakdown,
                "thermal": thermal_breakdown,
                "faults": fault_breakdown,
            },
        )

        return observation, reward, terminated, truncated, info

    # -----------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------

    def render(self) -> Dict[str, Any]:
        """
        Lightweight state dump for debugging or external renderer use.
        """
        return self.observation_builder.build_debug_dict(self.state)

    def request_manual_abort(self) -> None:
        self._manual_abort_requested = True

    def get_valid_actions(self) -> List[Action]:
        return get_valid_actions(self.state)

    def get_action_mask(self) -> List[int]:
        valid = set(self.get_valid_actions())
        return [1 if action in valid else 0 for action in self.action_order]

    # -----------------------------------------------------
    # Internal initialization
    # -----------------------------------------------------

    def _initialize_state(self, options: Dict[str, Any]) -> None:
        dt_s = float(options.get("dt_s", getattr(self.mission_config, "dt_s", 5.0)))

        self.state.time.dt_s = dt_s
        self.state.time.sim_time_s = 0.0
        self.state.time.step_count = 0
        self.state.time.orbit_index = 0

        self.state.mode = getattr(self.mission_config, "initial_mode", SpacecraftMode.BOOT)

        # Initial thermal/power/attitude conditions
        self.state.eps.battery_soc_pct = float(
            options.get("initial_battery_soc_pct", getattr(self.mission_config, "initial_battery_soc_pct", 70.0))
        )
        self.state.thermal.bus_temp_c = float(
            options.get("initial_bus_temp_c", getattr(self.mission_config, "initial_bus_temp_c", 18.0))
        )
        self.state.thermal.battery_temp_c = float(
            options.get("initial_battery_temp_c", getattr(self.mission_config, "initial_battery_temp_c", 12.0))
        )
        self.state.thermal.payload_temp_c = float(
            options.get("initial_payload_temp_c", getattr(self.mission_config, "initial_payload_temp_c", 14.0))
        )
        self.state.thermal.radio_temp_c = float(
            options.get("initial_radio_temp_c", getattr(self.mission_config, "initial_radio_temp_c", 16.0))
        )
        self.state.adcs.pointing_error_deg = float(
            options.get("initial_pointing_error_deg", getattr(self.mission_config, "initial_pointing_error_deg", 8.0))
        )
        self.state.adcs.body_rate_deg_s = float(
            options.get("initial_body_rate_deg_s", getattr(self.mission_config, "initial_body_rate_deg_s", 1.0))
        )
        self.state.adcs.wheel_momentum_nms = float(
            options.get("initial_wheel_momentum_nms", getattr(self.mission_config, "initial_wheel_momentum_nms", 0.03))
        )

        # Orbit init
        initial_phase = float(
            options.get("initial_orbit_phase", getattr(self.mission_config, "initial_orbit_phase", 0.02))
        )
        self.orbit.reset(
            self.state.orbit,
            initial_phase=initial_phase,
            orbit_index=0,
        )

        # Subsystem init
        self.power.initialize(self.state.eps)
        self.thermal.initialize(self.state)
        self.adcs.initialize(self.state)
        self.comms.initialize(self.state)
        self.cdh.initialize(self.state)
        self.payload.initialize(self.state)
        self.faults.initialize(self.state)

        # Start in safe sun acquire after boot
        self.state.adcs.mode = AttitudeMode.SAFE_SUN_ACQUIRE
        self.state.comms.mode = CommsMode.LISTEN
        self.state.cdh.mode = DataHandlingMode.IDLE
        self.state.payload.mode = PayloadMode.OFF

        self._post_step_mode_sanity()

    def _reset_rngs(self, seed: int) -> None:
        # Recreate stochastic subsystems with deterministic seeds
        self.orbit = OrbitPropagator(config=OrbitConfig(random_seed=seed))
        self.comms = CommsSubsystem(config=CommsConfig(random_seed=seed + 1))
        self.payload = PayloadSubsystem(config=PayloadConfig(random_seed=seed + 2))

    # -----------------------------------------------------
    # Step flow helpers
    # -----------------------------------------------------

    def _resolve_action(self, action: int | Action) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, int):
            return index_to_action(action)
        raise TypeError(f"Unsupported action type: {type(action)}")

    def _advance_time_and_orbit(self) -> None:
        old_phase = self.state.orbit.orbit_phase
        self.orbit.step(
            orbit_state=self.state.orbit,
            dt_s=self.state.time.dt_s,
            time_orbit_index=self.state.time.orbit_index,
        )
        if self.state.orbit.orbit_phase < old_phase:
            self.state.time.orbit_index += 1
            # keep orbit subsystem's internal continuity in sync
            self.orbit.step(
                orbit_state=self.state.orbit,
                dt_s=0.0,
                time_orbit_index=self.state.time.orbit_index,
            )
        self.state.time.advance()

    def _pre_step_mode_sanity(self, action: Action) -> None:
        # Boot should quickly transition to nominal-safe operations
        if self.state.mode == SpacecraftMode.BOOT:
            self.state.mode = SpacecraftMode.NOMINAL

        # Safe mode constraints
        if self.state.faults.safe_mode_latched:
            if self.state.mode not in (SpacecraftMode.SAFE, SpacecraftMode.FAULT_RECOVERY, SpacecraftMode.SURVIVAL):
                self.state.mode = SpacecraftMode.SAFE

        # Some actions imply explicit modes
        if action in (Action.DOWNLINK_LOW_RATE, Action.DOWNLINK_HIGH_RATE, Action.PREPARE_DOWNLINK):
            self.state.mode = SpacecraftMode.DOWNLINK
        elif action in (Action.CAPTURE_IMAGE, Action.RUN_CLASSIFIER, Action.STORE_FRAME, Action.DISCARD_FRAME):
            self.state.mode = SpacecraftMode.SCIENCE

    def _post_step_mode_sanity(self) -> None:
        # Survival if battery is truly catastrophic
        if self.state.eps.battery_soc_pct <= 5.0:
            self.state.mode = SpacecraftMode.SURVIVAL

        # Eclipse conserve
        if self.state.orbit.in_eclipse and self.state.eps.battery_soc_pct <= 30.0:
            if self.state.mode not in (SpacecraftMode.SAFE, SpacecraftMode.FAULT_RECOVERY, SpacecraftMode.SURVIVAL):
                self.state.mode = SpacecraftMode.ECLIPSE_POWER_SAVE

        # If faults latched safe mode, enforce it
        if self.state.faults.safe_mode_latched:
            if self.state.mode not in (SpacecraftMode.SAFE, SpacecraftMode.FAULT_RECOVERY, SpacecraftMode.SURVIVAL):
                self.state.mode = SpacecraftMode.SAFE

    # -----------------------------------------------------
    # Observation / info formatting
    # -----------------------------------------------------

    def _format_observation(self) -> Any:
        bundle = self.observation_builder.build_all(
            state=self.state,
            valid_actions=self.get_valid_actions(),
            ordered_actions=self.action_order,
        )

        obs_type = getattr(self.rl_config, "observation_type", "bundle")
        if obs_type == "tabular":
            return bundle["tabular_tuple"]
        if obs_type == "vector":
            return bundle["vector"]
        return bundle

    def _build_info_dict(
        self,
        done: bool,
        reward: float,
        extra_notes: Optional[List[str]] = None,
        subsystem_debug: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        debug = self.observation_builder.build_debug_dict(self.state)

        info: Dict[str, Any] = {
            "reward": reward,
            "done": done,
            "end_reason": self.state.end_reason.name,
            "valid_actions": [a.name for a in self.get_valid_actions()],
            "action_mask": self.get_action_mask(),
            "debug": debug,
            "reward_components": dict(self.state.step_reward.components),
            "reward_events": [e.name for e in self.state.step_reward.events],
            "faults_active": [
                {
                    "subsystem": f.subsystem.value,
                    "fault_type": f.fault_type.name,
                    "level": f.level.name,
                    "message": f.message,
                }
                for f in self.state.faults.active_faults
                if f.active
            ],
            "episode_stats": {
                "total_reward": self.state.episode_stats.total_reward,
                "useful_images_captured": self.state.episode_stats.useful_images_captured,
                "cloudy_images_captured": self.state.episode_stats.cloudy_images_captured,
                "useful_images_discarded": self.state.episode_stats.useful_images_discarded,
                "useful_images_stored": self.state.episode_stats.useful_images_stored,
                "useful_images_downlinked": self.state.episode_stats.useful_images_downlinked,
                "total_data_downlinked_mb": self.state.episode_stats.total_data_downlinked_mb,
                "safe_mode_entries": self.state.episode_stats.safe_mode_entries,
                "fault_count": self.state.episode_stats.fault_count,
            },
            "notes": list(extra_notes or []),
        }

        if self.last_action_validation is not None:
            info["last_action_validation"] = {
                "valid": self.last_action_validation.valid,
                "reasons": list(self.last_action_validation.reasons),
                "warnings": list(self.last_action_validation.warnings),
            }

        if subsystem_debug is not None:
            info["subsystems"] = subsystem_debug

        return info


# =========================================================
# Smoke-test helper
# =========================================================

def env_smoke_summary(
    steps: int = 25,
    action_sequence: Optional[List[Action]] = None,
) -> Dict[str, Any]:
    """
    Quick environment-level sanity test.

    Runs a short rollout and returns summary metrics.
    """
    env = CubeSatEnv()
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    done = False
    last_info: Dict[str, Any] = info

    if not action_sequence:
        action_sequence = [Action.NO_OP] * steps

    for i in range(min(steps, len(action_sequence))):
        _, reward, terminated, truncated, last_info = env.step(action_sequence[i])
        total_reward += reward
        done = terminated or truncated
        if done:
            break

    return {
        "steps_run": env.state.time.step_count,
        "orbit_index": env.state.time.orbit_index,
        "final_mode": env.state.mode.name,
        "battery_soc_pct": env.state.eps.battery_soc_pct,
        "memory_used_mb": env.state.cdh.memory_used_mb,
        "queue_mb": env.state.cdh.downlink_queue_mb,
        "active_fault_count": len([f for f in env.state.faults.active_faults if f.active]),
        "total_reward": total_reward,
        "done": done,
        "end_reason": env.state.end_reason.name,
        "last_notes": last_info.get("notes", []),
    }
