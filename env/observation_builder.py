from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.enums import (
    Action,
    BatteryBand,
    FaultLevel,
    FrameClass,
    GroundPassState,
    LinkQuality,
    MemoryPressure,
    PointingQuality,
    SpacecraftMode,
    SunlightState,
    TargetOpportunity,
    TemperatureBand,
)
from models.state_models import AgentObservation, SpacecraftState


# =========================================================
# Generic helpers
# =========================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    value = clamp(value, low, high)
    return (value - low) / (high - low)


def one_hot(index: int, size: int) -> List[float]:
    vec = [0.0] * size
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def bin_by_thresholds(value: float, thresholds: Sequence[float]) -> int:
    """
    Returns the bin index for a monotonically increasing threshold list.
    Example:
        thresholds=[10, 30, 70, 95]
        value=25 -> 1
        value=96 -> 4
    """
    for idx, threshold in enumerate(thresholds):
        if value < threshold:
            return idx
    return len(thresholds)


# =========================================================
# Observation config
# =========================================================

@dataclass(frozen=True)
class ObservationConfig:
    """
    Controls which observation variants are built and how they are quantized.
    """
    include_action_mask: bool = True
    include_debug_fields: bool = True

    # Queue / memory scaling
    queue_norm_max_mb: float = 2048.0

    # Continuous-to-discrete thresholds
    battery_soc_thresholds_pct: Tuple[float, ...] = (10.0, 30.0, 70.0, 95.0)
    battery_temp_thresholds_c: Tuple[float, ...] = (0.0, 5.0, 35.0, 45.0)
    payload_temp_thresholds_c: Tuple[float, ...] = (-5.0, 0.0, 40.0, 45.0)
    pointing_error_thresholds_deg: Tuple[float, ...] = (0.3, 1.5, 5.0)
    sun_elevation_thresholds_deg: Tuple[float, ...] = (0.0, 10.0, 25.0, 45.0)
    gs_elevation_thresholds_deg: Tuple[float, ...] = (0.0, 10.0, 30.0, 60.0)
    wheel_momentum_util_thresholds: Tuple[float, ...] = (0.4, 0.7, 0.9)
    classifier_conf_thresholds: Tuple[float, ...] = (0.2, 0.5, 0.8)
    usefulness_thresholds: Tuple[float, ...] = (0.2, 0.5, 0.8)

    # Optional one-hot controls for DQN feature expansion
    use_one_hot_for_categoricals: bool = False


DEFAULT_OBSERVATION_CONFIG = ObservationConfig()


# =========================================================
# Discrete observation structures
# =========================================================

@dataclass(frozen=True)
class TabularObservation:
    """
    Compact discrete observation for tabular RL methods such as Q-learning.

    Each field should have a small finite cardinality.
    """
    spacecraft_mode: int
    sunlight_state: int
    target_opportunity: int
    battery_band: int
    battery_temp_band: int
    memory_pressure: int
    has_frame: int
    frame_class: int
    pointing_quality: int
    ground_pass_state: int
    link_quality: int
    safe_mode_latched: int
    highest_fault_level: int

    def as_tuple(self) -> Tuple[int, ...]:
        return (
            self.spacecraft_mode,
            self.sunlight_state,
            self.target_opportunity,
            self.battery_band,
            self.battery_temp_band,
            self.memory_pressure,
            self.has_frame,
            self.frame_class,
            self.pointing_quality,
            self.ground_pass_state,
            self.link_quality,
            self.safe_mode_latched,
            self.highest_fault_level,
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "spacecraft_mode": self.spacecraft_mode,
            "sunlight_state": self.sunlight_state,
            "target_opportunity": self.target_opportunity,
            "battery_band": self.battery_band,
            "battery_temp_band": self.battery_temp_band,
            "memory_pressure": self.memory_pressure,
            "has_frame": self.has_frame,
            "frame_class": self.frame_class,
            "pointing_quality": self.pointing_quality,
            "ground_pass_state": self.ground_pass_state,
            "link_quality": self.link_quality,
            "safe_mode_latched": self.safe_mode_latched,
            "highest_fault_level": self.highest_fault_level,
        }


@dataclass(frozen=True)
class RichDiscreteObservation:
    """
    Richer discrete observation for debugging or more expressive non-tabular agents.
    """
    orbit_phase_bin: int
    sun_elevation_bin: int
    gs_elevation_bin: int
    battery_soc_bin: int
    battery_temp_bin: int
    payload_temp_bin: int
    memory_pressure: int
    queue_bin: int
    pointing_error_bin: int
    wheel_momentum_bin: int
    target_opportunity: int
    frame_class: int
    classifier_conf_bin: int
    usefulness_bin: int
    ground_pass_state: int
    link_quality: int
    fault_level: int

    def as_tuple(self) -> Tuple[int, ...]:
        return (
            self.orbit_phase_bin,
            self.sun_elevation_bin,
            self.gs_elevation_bin,
            self.battery_soc_bin,
            self.battery_temp_bin,
            self.payload_temp_bin,
            self.memory_pressure,
            self.queue_bin,
            self.pointing_error_bin,
            self.wheel_momentum_bin,
            self.target_opportunity,
            self.frame_class,
            self.classifier_conf_bin,
            self.usefulness_bin,
            self.ground_pass_state,
            self.link_quality,
            self.fault_level,
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "orbit_phase_bin": self.orbit_phase_bin,
            "sun_elevation_bin": self.sun_elevation_bin,
            "gs_elevation_bin": self.gs_elevation_bin,
            "battery_soc_bin": self.battery_soc_bin,
            "battery_temp_bin": self.battery_temp_bin,
            "payload_temp_bin": self.payload_temp_bin,
            "memory_pressure": self.memory_pressure,
            "queue_bin": self.queue_bin,
            "pointing_error_bin": self.pointing_error_bin,
            "wheel_momentum_bin": self.wheel_momentum_bin,
            "target_opportunity": self.target_opportunity,
            "frame_class": self.frame_class,
            "classifier_conf_bin": self.classifier_conf_bin,
            "usefulness_bin": self.usefulness_bin,
            "ground_pass_state": self.ground_pass_state,
            "link_quality": self.link_quality,
            "fault_level": self.fault_level,
        }


# =========================================================
# Observation builder
# =========================================================

class ObservationBuilder:
    """
    Converts the full spacecraft simulator state into multiple agent-facing views.

    Supported observation products:
    - compact tabular tuple for Q-learning
    - richer discrete tuple for interpretable policies
    - normalized vector for DQN / PPO / neural agents
    - debug dictionary for logs / renderer / tests
    """

    def __init__(self, config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG):
        self.config = config

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def build_all(
        self,
        state: SpacecraftState,
        valid_actions: Optional[Sequence[Action]] = None,
        ordered_actions: Optional[Sequence[Action]] = None,
    ) -> Dict[str, Any]:
        """
        Build all common observation forms at once.
        """
        tabular = self.build_tabular_observation(state)
        rich_discrete = self.build_rich_discrete_observation(state)
        agent_obs = self.build_agent_observation(state)
        vector = self.build_dqn_vector(state)
        debug = self.build_debug_dict(state)

        payload: Dict[str, Any] = {
            "tabular": tabular,
            "tabular_tuple": tabular.as_tuple(),
            "rich_discrete": rich_discrete,
            "rich_discrete_tuple": rich_discrete.as_tuple(),
            "agent_observation": agent_obs,
            "vector": vector,
            "debug": debug,
        }

        if self.config.include_action_mask and ordered_actions is not None:
            payload["action_mask"] = self.build_action_mask(
                valid_actions=valid_actions,
                ordered_actions=ordered_actions,
            )

        return payload

    def build_tabular_observation(self, state: SpacecraftState) -> TabularObservation:
        """
        Small-cardinality observation intended for classic tabular RL.
        """
        return TabularObservation(
            spacecraft_mode=int(state.mode),
            sunlight_state=int(state.orbit.sunlight_state),
            target_opportunity=int(state.orbit.target_opportunity),
            battery_band=int(self._derive_battery_band(state)),
            battery_temp_band=int(self._derive_battery_temp_band(state)),
            memory_pressure=int(state.cdh.memory_pressure),
            has_frame=int(state.payload.has_frame),
            frame_class=int(state.payload.current_frame_class),
            pointing_quality=int(self._derive_pointing_quality(state)),
            ground_pass_state=int(self._derive_ground_pass_state(state)),
            link_quality=int(self._derive_link_quality(state)),
            safe_mode_latched=int(state.faults.safe_mode_latched),
            highest_fault_level=int(state.faults.highest_fault_level),
        )

    def build_rich_discrete_observation(self, state: SpacecraftState) -> RichDiscreteObservation:
        """
        Higher-resolution discrete observation for debugging or more capable agents.
        """
        return RichDiscreteObservation(
            orbit_phase_bin=self._orbit_phase_bin(state.orbit.orbit_phase),
            sun_elevation_bin=bin_by_thresholds(
                state.orbit.sun_elevation_deg,
                self.config.sun_elevation_thresholds_deg,
            ),
            gs_elevation_bin=bin_by_thresholds(
                state.comms.gs_elevation_deg,
                self.config.gs_elevation_thresholds_deg,
            ),
            battery_soc_bin=bin_by_thresholds(
                state.eps.battery_soc_pct,
                self.config.battery_soc_thresholds_pct,
            ),
            battery_temp_bin=bin_by_thresholds(
                state.thermal.battery_temp_c,
                self.config.battery_temp_thresholds_c,
            ),
            payload_temp_bin=bin_by_thresholds(
                state.thermal.payload_temp_c,
                self.config.payload_temp_thresholds_c,
            ),
            memory_pressure=int(state.cdh.memory_pressure),
            queue_bin=self._queue_bin(state),
            pointing_error_bin=bin_by_thresholds(
                state.adcs.pointing_error_deg,
                self.config.pointing_error_thresholds_deg,
            ),
            wheel_momentum_bin=self._wheel_momentum_bin(state),
            target_opportunity=int(state.orbit.target_opportunity),
            frame_class=int(state.payload.current_frame_class),
            classifier_conf_bin=bin_by_thresholds(
                state.payload.classifier_confidence,
                self.config.classifier_conf_thresholds,
            ),
            usefulness_bin=bin_by_thresholds(
                state.payload.current_frame_usefulness,
                self.config.usefulness_thresholds,
            ),
            ground_pass_state=int(self._derive_ground_pass_state(state)),
            link_quality=int(self._derive_link_quality(state)),
            fault_level=int(state.faults.highest_fault_level),
        )

    def build_agent_observation(self, state: SpacecraftState) -> AgentObservation:
        """
        Reuses the structured normalized observation defined in state_models.py.
        """
        return AgentObservation.from_spacecraft_state(state)

    def build_dqn_vector(self, state: SpacecraftState) -> List[float]:
        """
        Build a neural-network-friendly feature vector.

        By default this uses a compact normalized representation.
        Optionally, categorical variables can be expanded with one-hot encoding.
        """
        if self.config.use_one_hot_for_categoricals:
            return self._build_dqn_vector_one_hot(state)
        return self._build_dqn_vector_compact(state)

    def build_debug_dict(self, state: SpacecraftState) -> Dict[str, Any]:
        """
        High-readability dictionary for logs, tests, dashboards, and the renderer.
        """
        debug: Dict[str, Any] = {
            "spacecraft_mode": state.mode.name,
            "time_step": state.time.step_count,
            "sim_time_s": state.time.sim_time_s,
            "sim_time_min": state.time.sim_time_min,
            "orbit_index": state.time.orbit_index,
            "orbit_phase": state.orbit.orbit_phase,
            "sunlight_state": state.orbit.sunlight_state.name,
            "sun_elevation_deg": state.orbit.sun_elevation_deg,
            "over_target": state.orbit.over_target,
            "target_opportunity": state.orbit.target_opportunity.name,
            "battery_soc_pct": state.eps.battery_soc_pct,
            "battery_band": self._derive_battery_band(state).name,
            "battery_temp_c": state.thermal.battery_temp_c,
            "battery_temp_band": self._derive_battery_temp_band(state).name,
            "solar_input_w": state.eps.solar_input_w,
            "load_output_w": state.eps.load_output_w,
            "net_power_w": state.eps.net_power_w,
            "power_positive": state.power_positive,
            "attitude_mode": state.adcs.mode.name,
            "pointing_error_deg": state.adcs.pointing_error_deg,
            "pointing_quality": self._derive_pointing_quality(state).name,
            "wheel_momentum_nms": state.adcs.wheel_momentum_nms,
            "wheel_momentum_utilization": self._wheel_momentum_utilization(state),
            "payload_mode": state.payload.mode.name,
            "has_frame": state.payload.has_frame,
            "frame_class": state.payload.current_frame_class.name,
            "frame_cloud_prob": state.payload.current_frame_cloud_prob,
            "frame_usefulness": state.payload.current_frame_usefulness,
            "classifier_confidence": state.payload.classifier_confidence,
            "memory_used_mb": state.cdh.memory_used_mb,
            "memory_free_mb": state.cdh.memory_free_mb,
            "memory_pressure": state.cdh.memory_pressure.name,
            "downlink_queue_mb": state.cdh.downlink_queue_mb,
            "comms_mode": state.comms.mode.name,
            "gs_visible": state.comms.gs_visible,
            "ground_pass_state": self._derive_ground_pass_state(state).name,
            "link_quality": self._derive_link_quality(state).name,
            "gs_elevation_deg": state.comms.gs_elevation_deg,
            "highest_fault_level": state.faults.highest_fault_level.name,
            "last_fault": state.faults.last_fault.name,
            "safe_mode_latched": state.faults.safe_mode_latched,
            "alive": state.alive,
            "done": state.done,
            "end_reason": state.end_reason.name,
            "last_action": state.last_action.name,
            "previous_action": state.previous_action.name,
        }

        if self.config.include_debug_fields:
            debug.update(
                {
                    "battery_soc_bin": bin_by_thresholds(
                        state.eps.battery_soc_pct,
                        self.config.battery_soc_thresholds_pct,
                    ),
                    "queue_bin": self._queue_bin(state),
                    "orbit_phase_bin": self._orbit_phase_bin(state.orbit.orbit_phase),
                    "pointing_error_bin": bin_by_thresholds(
                        state.adcs.pointing_error_deg,
                        self.config.pointing_error_thresholds_deg,
                    ),
                }
            )

        return debug

    def build_action_mask(
        self,
        valid_actions: Optional[Sequence[Action]],
        ordered_actions: Sequence[Action],
    ) -> List[int]:
        """
        Build a binary action mask aligned to ordered_actions.
        """
        valid_set = set(valid_actions or [])
        return [1 if action in valid_set else 0 for action in ordered_actions]

    # -----------------------------------------------------
    # Compact DQN vector
    # -----------------------------------------------------

    def _build_dqn_vector_compact(self, state: SpacecraftState) -> List[float]:
        battery_band = float(int(self._derive_battery_band(state)))
        battery_temp_band = float(int(self._derive_battery_temp_band(state)))
        pointing_quality = float(int(self._derive_pointing_quality(state)))
        ground_pass_state = float(int(self._derive_ground_pass_state(state)))
        link_quality = float(int(self._derive_link_quality(state)))
        memory_pressure = float(int(state.cdh.memory_pressure))

        queue_norm = normalize(
            state.cdh.downlink_queue_mb,
            0.0,
            max(1.0, self.config.queue_norm_max_mb),
        )
        wheel_util = self._wheel_momentum_utilization(state)

        return [
            # Mission / geometry
            normalize(state.orbit.orbit_phase, 0.0, 1.0),
            float(int(state.orbit.sunlight_state)),
            normalize(state.orbit.sun_elevation_deg, -30.0, 90.0),
            float(state.orbit.over_target),
            float(int(state.orbit.target_opportunity)),
            # EPS / thermal
            normalize(state.eps.battery_soc_pct, 0.0, 100.0),
            battery_band,
            normalize(state.thermal.battery_temp_c, -20.0, 60.0),
            battery_temp_band,
            normalize(state.eps.solar_input_w, 0.0, 25.0),
            normalize(state.eps.load_output_w, 0.0, 25.0),
            normalize(state.eps.net_power_w, -20.0, 20.0),
            float(state.power_positive),
            # ADCS
            float(int(state.adcs.mode)),
            normalize(state.adcs.pointing_error_deg, 0.0, 10.0),
            pointing_quality,
            wheel_util,
            float(state.adcs.wheels_saturated),
            # Payload / classifier
            float(int(state.payload.mode)),
            float(state.payload.has_frame),
            float(int(state.payload.current_frame_class)),
            clamp(state.payload.current_frame_cloud_prob, 0.0, 1.0),
            clamp(state.payload.current_frame_usefulness, 0.0, 1.0),
            clamp(state.payload.classifier_confidence, 0.0, 1.0),
            # CDH
            normalize(state.cdh.memory_utilization, 0.0, 1.0),
            memory_pressure,
            queue_norm,
            # Comms
            float(int(state.comms.mode)),
            float(state.comms.gs_visible),
            normalize(state.comms.gs_elevation_deg, 0.0, 90.0),
            ground_pass_state,
            link_quality,
            # FDIR
            float(state.faults.safe_mode_latched),
            float(int(state.faults.highest_fault_level)),
            float(state.critical_fault),
            # Top-level
            float(int(state.mode)),
        ]

    def _build_dqn_vector_one_hot(self, state: SpacecraftState) -> List[float]:
        """
        One-hot-expanded variant for neural agents that benefit from explicit categorical separation.
        """
        features: List[float] = []

        # Continuous
        features.extend(
            [
                normalize(state.orbit.orbit_phase, 0.0, 1.0),
                normalize(state.orbit.sun_elevation_deg, -30.0, 90.0),
                normalize(state.eps.battery_soc_pct, 0.0, 100.0),
                normalize(state.thermal.battery_temp_c, -20.0, 60.0),
                normalize(state.eps.solar_input_w, 0.0, 25.0),
                normalize(state.eps.load_output_w, 0.0, 25.0),
                normalize(state.eps.net_power_w, -20.0, 20.0),
                normalize(state.adcs.pointing_error_deg, 0.0, 10.0),
                self._wheel_momentum_utilization(state),
                normalize(state.cdh.memory_utilization, 0.0, 1.0),
                normalize(
                    state.cdh.downlink_queue_mb,
                    0.0,
                    max(1.0, self.config.queue_norm_max_mb),
                ),
                normalize(state.comms.gs_elevation_deg, 0.0, 90.0),
                clamp(state.payload.current_frame_cloud_prob, 0.0, 1.0),
                clamp(state.payload.current_frame_usefulness, 0.0, 1.0),
                clamp(state.payload.classifier_confidence, 0.0, 1.0),
            ]
        )

        # Binary flags
        features.extend(
            [
                float(int(state.orbit.sunlight_state == SunlightState.SUNLIT)),
                float(state.orbit.over_target),
                float(state.power_positive),
                float(state.adcs.wheels_saturated),
                float(state.payload.has_frame),
                float(state.comms.gs_visible),
                float(state.faults.safe_mode_latched),
                float(state.critical_fault),
            ]
        )

        # One-hot categoricals
        features.extend(one_hot(int(state.mode), self._enum_cardinality(SpacecraftMode)))
        features.extend(one_hot(int(self._derive_battery_band(state)), self._enum_cardinality(BatteryBand)))
        features.extend(one_hot(int(self._derive_battery_temp_band(state)), self._enum_cardinality(TemperatureBand)))
        features.extend(one_hot(int(state.cdh.memory_pressure), self._enum_cardinality(MemoryPressure)))
        features.extend(one_hot(int(state.adcs.mode), self._enum_cardinality(type(state.adcs.mode))))
        features.extend(one_hot(int(self._derive_pointing_quality(state)), self._enum_cardinality(PointingQuality)))
        features.extend(one_hot(int(state.payload.mode), self._enum_cardinality(type(state.payload.mode))))
        features.extend(one_hot(int(state.payload.current_frame_class), self._enum_cardinality(FrameClass)))
        features.extend(one_hot(int(state.comms.mode), self._enum_cardinality(type(state.comms.mode))))
        features.extend(one_hot(int(self._derive_ground_pass_state(state)), self._enum_cardinality(GroundPassState)))
        features.extend(one_hot(int(self._derive_link_quality(state)), self._enum_cardinality(LinkQuality)))
        features.extend(one_hot(int(state.orbit.target_opportunity), self._enum_cardinality(TargetOpportunity)))
        features.extend(one_hot(int(state.faults.highest_fault_level), self._enum_cardinality(FaultLevel)))

        return features

    # -----------------------------------------------------
    # Derived-state helpers
    # -----------------------------------------------------

    def _derive_battery_band(self, state: SpacecraftState) -> BatteryBand:
        """
        Prefer the EPS-derived band if present and valid.
        """
        try:
            return state.eps.battery_band
        except AttributeError:
            idx = bin_by_thresholds(
                state.eps.battery_soc_pct,
                self.config.battery_soc_thresholds_pct,
            )
            return BatteryBand(idx)

    def _derive_battery_temp_band(self, state: SpacecraftState) -> TemperatureBand:
        try:
            return state.thermal.battery_temp_band
        except AttributeError:
            idx = bin_by_thresholds(
                state.thermal.battery_temp_c,
                self.config.battery_temp_thresholds_c,
            )
            return TemperatureBand(idx)

    def _derive_pointing_quality(self, state: SpacecraftState) -> PointingQuality:
        try:
            return state.adcs.pointing_quality
        except AttributeError:
            err = state.adcs.pointing_error_deg
            if err > 5.0:
                return PointingQuality.INVALID
            if err > 1.5:
                return PointingQuality.COARSE
            if err > 0.3:
                return PointingQuality.USABLE
            return PointingQuality.PRECISE

    def _derive_ground_pass_state(self, state: SpacecraftState) -> GroundPassState:
        """
        Uses comms pass state if available; otherwise derives a reasonable fallback.
        """
        try:
            return state.comms.pass_state
        except AttributeError:
            if not state.comms.gs_visible:
                return GroundPassState.NONE

            el = state.comms.gs_elevation_deg
            if el < 5.0:
                return GroundPassState.ACQUIRE
            if el < 20.0:
                return GroundPassState.LOW_ELEVATION
            if el < 50.0:
                return GroundPassState.MID_ELEVATION
            return GroundPassState.HIGH_ELEVATION

    def _derive_link_quality(self, state: SpacecraftState) -> LinkQuality:
        try:
            return state.comms.link_quality
        except AttributeError:
            if not state.comms.gs_visible:
                return LinkQuality.NONE

            el = state.comms.gs_elevation_deg
            if el < 10.0:
                return LinkQuality.POOR
            if el < 25.0:
                return LinkQuality.MARGINAL
            if el < 55.0:
                return LinkQuality.GOOD
            return LinkQuality.EXCELLENT

    def _wheel_momentum_utilization(self, state: SpacecraftState) -> float:
        limit = max(1e-6, state.adcs.wheel_momentum_limit_nms)
        return clamp(state.adcs.wheel_momentum_nms / limit, 0.0, 1.5)

    def _wheel_momentum_bin(self, state: SpacecraftState) -> int:
        return bin_by_thresholds(
            self._wheel_momentum_utilization(state),
            self.config.wheel_momentum_util_thresholds,
        )

    def _queue_bin(self, state: SpacecraftState) -> int:
        """
        Queue bins are defined by fraction of available storage, not fixed MB,
        making them robust to different mission memory budgets.
        """
        capacity = max(1.0, state.cdh.memory_capacity_mb)
        frac = clamp(state.cdh.downlink_queue_mb / capacity, 0.0, 1.5)

        if frac == 0.0:
            return 0
        if frac < 0.10:
            return 1
        if frac < 0.30:
            return 2
        if frac < 0.60:
            return 3
        return 4

    def _orbit_phase_bin(self, orbit_phase: float) -> int:
        """
        Quarter-orbit binning.
        """
        phase = orbit_phase % 1.0
        if phase < 0.25:
            return 0
        if phase < 0.50:
            return 1
        if phase < 0.75:
            return 2
        return 3

    @staticmethod
    def _enum_cardinality(enum_cls: Any) -> int:
        return len(list(enum_cls))


# =========================================================
# Convenience functions
# =========================================================

def build_default_observation_bundle(
    state: SpacecraftState,
    valid_actions: Optional[Sequence[Action]] = None,
    ordered_actions: Optional[Sequence[Action]] = None,
    config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
) -> Dict[str, Any]:
    builder = ObservationBuilder(config=config)
    return builder.build_all(
        state=state,
        valid_actions=valid_actions,
        ordered_actions=ordered_actions,
    )


def build_tabular_state_tuple(
    state: SpacecraftState,
    config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
) -> Tuple[int, ...]:
    builder = ObservationBuilder(config=config)
    return builder.build_tabular_observation(state).as_tuple()


def build_dqn_vector(
    state: SpacecraftState,
    config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
) -> List[float]:
    builder = ObservationBuilder(config=config)
    return builder.build_dqn_vector(state)