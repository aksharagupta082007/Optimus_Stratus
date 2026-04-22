from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

from .enums import (
    Action,
    AttitudeMode,
    BatteryBand,
    CommsMode,
    DataHandlingMode,
    EpisodeEndReason,
    FaultLevel,
    FaultType,
    FrameClass,
    GroundPassState,
    LinkQuality,
    MemoryPressure,
    PayloadMode,
    PointingQuality,
    PowerMode,
    ResetCause,
    RewardEvent,
    SpacecraftMode,
    SubsystemName,
    SunlightState,
    TargetOpportunity,
    TemperatureBand,
    ThermalMode,
)


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


# =========================================================
# Time / orbital context
# =========================================================

@dataclass
class SimTime:
    """
    Simulation clock.
    All times are expressed in seconds for consistency.
    """
    dt_s: float = 5.0
    sim_time_s: float = 0.0
    step_count: int = 0
    orbit_index: int = 0

    def advance(self) -> None:
        self.sim_time_s += self.dt_s
        self.step_count += 1

    @property
    def sim_time_min(self) -> float:
        return self.sim_time_s / 60.0

    @property
    def sim_time_hr(self) -> float:
        return self.sim_time_s / 3600.0


@dataclass
class OrbitState:
    """
    Simplified operational orbit state for a LEO Earth-observation CubeSat.
    This is not a full propagator state vector; it is a mission-simulator state.
    """
    orbit_period_s: float = 5400.0
    altitude_km: float = 525.0
    inclination_deg: float = 97.5

    orbit_phase: float = 0.0                 # [0, 1)
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    beta_angle_deg: float = 20.0

    sunlight_state: SunlightState = SunlightState.SUNLIT
    sun_elevation_deg: float = 45.0

    over_target: bool = False
    target_opportunity: TargetOpportunity = TargetOpportunity.NONE
    land_fraction_hint: float = 0.5

    def advance(self, dt_s: float) -> None:
        """
        Advance orbit phase only. A more detailed orbit module can overwrite
        latitude/longitude/sunlight externally.
        """
        if self.orbit_period_s <= 0:
            return

        delta = dt_s / self.orbit_period_s
        self.orbit_phase = (self.orbit_phase + delta) % 1.0

        # Track orbit count on wrap
        # Caller can inspect wrap separately if needed.

    @property
    def in_eclipse(self) -> bool:
        return self.sunlight_state == SunlightState.ECLIPSE

    @property
    def is_sunlit(self) -> bool:
        return self.sunlight_state == SunlightState.SUNLIT

    @property
    def day_imaging_valid(self) -> bool:
        return self.sun_elevation_deg >= 10.0 and self.is_sunlit


# =========================================================
# Subsystem states
# =========================================================

@dataclass
class EPSState:
    """
    Electrical Power Subsystem state.
    """
    mode: PowerMode = PowerMode.NOMINAL

    battery_soc_pct: float = 80.0
    battery_capacity_wh: float = 30.0
    battery_voltage_v: float = 7.6
    battery_current_a: float = 0.0

    solar_input_w: float = 8.0
    load_output_w: float = 5.0
    net_power_w: float = 3.0

    battery_temp_c: float = 15.0
    heater_enabled: bool = False

    eps_power_positive: bool = True
    brownout_risk: bool = False

    min_soc_pct: float = 5.0
    max_soc_pct: float = 100.0

    def update_net_power(self) -> None:
        self.net_power_w = self.solar_input_w - self.load_output_w
        self.eps_power_positive = self.net_power_w >= 0.0
        self.brownout_risk = self.battery_soc_pct <= self.min_soc_pct

    @property
    def battery_band(self) -> BatteryBand:
        soc = self.battery_soc_pct
        if soc < 10:
            return BatteryBand.CRITICAL
        if soc < 30:
            return BatteryBand.LOW
        if soc < 70:
            return BatteryBand.MEDIUM
        if soc < 95:
            return BatteryBand.HIGH
        return BatteryBand.FULL

    @property
    def battery_temp_band(self) -> TemperatureBand:
        t = self.battery_temp_c
        if t < -5:
            return TemperatureBand.TOO_COLD
        if t < 5:
            return TemperatureBand.COLD
        if t <= 30:
            return TemperatureBand.NOMINAL
        if t <= 40:
            return TemperatureBand.WARM
        return TemperatureBand.TOO_HOT


@dataclass
class ThermalState:
    """
    High-level thermal state.
    """
    mode: ThermalMode = ThermalMode.NOMINAL

    bus_temp_c: float = 20.0
    battery_temp_c: float = 15.0
    payload_temp_c: float = 18.0
    radio_temp_c: float = 20.0

    battery_heater_on: bool = False
    payload_heater_on: bool = False

    bus_temp_min_c: float = -10.0
    bus_temp_max_c: float = 50.0
    battery_temp_min_c: float = 0.0
    battery_temp_max_c: float = 35.0
    payload_temp_min_c: float = -5.0
    payload_temp_max_c: float = 40.0

    @property
    def battery_temp_band(self) -> TemperatureBand:
        t = self.battery_temp_c
        if t < self.battery_temp_min_c - 5:
            return TemperatureBand.TOO_COLD
        if t < self.battery_temp_min_c:
            return TemperatureBand.COLD
        if t <= self.battery_temp_max_c:
            return TemperatureBand.NOMINAL
        if t <= self.battery_temp_max_c + 5:
            return TemperatureBand.WARM
        return TemperatureBand.TOO_HOT

    @property
    def payload_temp_band(self) -> TemperatureBand:
        t = self.payload_temp_c
        if t < self.payload_temp_min_c - 5:
            return TemperatureBand.TOO_COLD
        if t < self.payload_temp_min_c:
            return TemperatureBand.COLD
        if t <= self.payload_temp_max_c:
            return TemperatureBand.NOMINAL
        if t <= self.payload_temp_max_c + 5:
            return TemperatureBand.WARM
        return TemperatureBand.TOO_HOT

    @property
    def thermal_violation(self) -> bool:
        return any(
            [
                self.bus_temp_c < self.bus_temp_min_c,
                self.bus_temp_c > self.bus_temp_max_c,
                self.battery_temp_c < self.battery_temp_min_c,
                self.battery_temp_c > self.battery_temp_max_c,
                self.payload_temp_c < self.payload_temp_min_c,
                self.payload_temp_c > self.payload_temp_max_c,
            ]
        )


@dataclass
class ADCSState:
    """
    Attitude Determination and Control System state.
    """
    mode: AttitudeMode = AttitudeMode.SAFE_SUN_ACQUIRE

    pointing_error_deg: float = 10.0
    body_rate_deg_s: float = 2.0
    wheel_momentum_nms: float = 0.05
    wheel_momentum_limit_nms: float = 0.12

    sun_vector_locked: bool = False
    nadir_locked: bool = False
    ground_track_locked: bool = False

    slew_time_remaining_s: float = 0.0
    target_body_frame_az_deg: float = 0.0
    target_body_frame_el_deg: float = 0.0

    star_tracker_available: bool = True
    imu_available: bool = True
    magnetometer_available: bool = True

    @property
    def wheels_saturated(self) -> bool:
        return self.wheel_momentum_nms >= self.wheel_momentum_limit_nms

    @property
    def pointing_quality(self) -> PointingQuality:
        e = self.pointing_error_deg
        if e > 5.0:
            return PointingQuality.INVALID
        if e > 1.5:
            return PointingQuality.COARSE
        if e > 0.3:
            return PointingQuality.USABLE
        return PointingQuality.PRECISE

    @property
    def settled(self) -> bool:
        return self.slew_time_remaining_s <= 0.0 and self.pointing_error_deg <= 1.5


@dataclass
class PayloadState:
    """
    Payload / imager / onboard inference state.
    """
    mode: PayloadMode = PayloadMode.OFF

    payload_enabled: bool = False
    imaging_ready: bool = False
    imaging_cooldown_s: float = 0.0
    warmup_time_remaining_s: float = 0.0

    current_frame_id: Optional[int] = None
    current_frame_size_mb: float = 0.0
    current_frame_cloud_prob: float = 0.0
    current_frame_usefulness: float = 0.0
    current_frame_class: FrameClass = FrameClass.NONE
    # Raw TFRecord bytes stored on CAPTURE_IMAGE; consumed by RUN_CLASSIFIER.
    # Excluded from RL observation; never serialised to disk.
    current_frame_tfrecord_bytes: Optional[bytes] = None

    classifier_confidence: float = 0.0
    classifier_last_latency_s: float = 0.0
    classifier_success: bool = True

    total_frames_captured: int = 0
    total_frames_useful: int = 0
    total_frames_discarded: int = 0

    def clear_current_frame(self) -> None:
        self.current_frame_id = None
        self.current_frame_size_mb = 0.0
        self.current_frame_cloud_prob = 0.0
        self.current_frame_usefulness = 0.0
        self.current_frame_class = FrameClass.NONE
        self.current_frame_tfrecord_bytes = None
        self.classifier_confidence = 0.0
        self.classifier_last_latency_s = 0.0
        self.classifier_success = True

    @property
    def has_frame(self) -> bool:
        return self.current_frame_id is not None


@dataclass
class CommsState:
    """
    Communications subsystem state.
    """
    mode: CommsMode = CommsMode.OFF

    gs_visible: bool = False
    pass_state: GroundPassState = GroundPassState.NONE
    link_quality: LinkQuality = LinkQuality.NONE
    gs_elevation_deg: float = 0.0
    range_km: float = 0.0

    low_rate_mbps: float = 0.25
    high_rate_mbps: float = 2.0
    current_rate_mbps: float = 0.0

    tx_enabled: bool = False
    rx_enabled: bool = False

    bytes_downlinked_this_step: int = 0
    total_bytes_downlinked: int = 0

    packet_success_prob: float = 1.0
    pass_time_remaining_s: float = 0.0

    @property
    def link_available(self) -> bool:
        return self.gs_visible and self.link_quality != LinkQuality.NONE

    @property
    def rate_mb_per_s(self) -> float:
        return self.current_rate_mbps / 8.0


@dataclass
class CDHState:
    """
    Command and Data Handling state.
    """
    mode: DataHandlingMode = DataHandlingMode.IDLE

    memory_capacity_mb: float = 4096.0
    memory_used_mb: float = 0.0
    downlink_queue_mb: float = 0.0

    raw_buffer_mb: float = 0.0
    processed_buffer_mb: float = 0.0

    filesystem_healthy: bool = True
    storage_corrupted: bool = False

    compression_ratio: float = 0.5
    cpu_load_pct: float = 10.0

    last_reset_cause: ResetCause = ResetCause.NONE
    reset_count: int = 0

    total_useful_mb_stored: float = 0.0
    total_useful_mb_downlinked: float = 0.0

    @property
    def memory_free_mb(self) -> float:
        return max(0.0, self.memory_capacity_mb - self.memory_used_mb)

    @property
    def memory_pressure(self) -> MemoryPressure:
        frac = 0.0 if self.memory_capacity_mb <= 0 else self.memory_used_mb / self.memory_capacity_mb
        if frac < 0.50:
            return MemoryPressure.LOW
        if frac < 0.75:
            return MemoryPressure.MEDIUM
        if frac < 0.95:
            return MemoryPressure.HIGH
        return MemoryPressure.CRITICAL

    @property
    def memory_utilization(self) -> float:
        if self.memory_capacity_mb <= 0:
            return 0.0
        return clamp(self.memory_used_mb / self.memory_capacity_mb, 0.0, 1.0)


@dataclass
class FaultRecord:
    subsystem: SubsystemName
    fault_type: FaultType
    level: FaultLevel
    message: str = ""
    active: bool = True
    first_seen_step: int = 0
    last_seen_step: int = 0


@dataclass
class FaultState:
    """
    Aggregated fault management state.
    """
    active_faults: List[FaultRecord] = field(default_factory=list)
    last_fault: FaultType = FaultType.NONE
    highest_fault_level: FaultLevel = FaultLevel.NONE
    safe_mode_latched: bool = False
    watchdog_triggered: bool = False

    def add_fault(
        self,
        subsystem: SubsystemName,
        fault_type: FaultType,
        level: FaultLevel,
        step_count: int,
        message: str = "",
    ) -> None:
        for fault in self.active_faults:
            if (
                fault.subsystem == subsystem
                and fault.fault_type == fault_type
                and fault.active
            ):
                fault.level = max(fault.level, level)
                fault.last_seen_step = step_count
                if message:
                    fault.message = message
                self.last_fault = fault_type
                self.highest_fault_level = max(self.highest_fault_level, level)
                return

        self.active_faults.append(
            FaultRecord(
                subsystem=subsystem,
                fault_type=fault_type,
                level=level,
                message=message,
                active=True,
                first_seen_step=step_count,
                last_seen_step=step_count,
            )
        )
        self.last_fault = fault_type
        self.highest_fault_level = max(self.highest_fault_level, level)

    def clear_fault(self, subsystem: SubsystemName, fault_type: FaultType) -> None:
        for fault in self.active_faults:
            if fault.subsystem == subsystem and fault.fault_type == fault_type and fault.active:
                fault.active = False
        self._recompute()

    def clear_all(self) -> None:
        for fault in self.active_faults:
            fault.active = False
        self._recompute()

    def has_fault(self, fault_type: FaultType) -> bool:
        return any(f.active and f.fault_type == fault_type for f in self.active_faults)

    def has_level_at_least(self, level: FaultLevel) -> bool:
        return any(f.active and f.level >= level for f in self.active_faults)

    def _recompute(self) -> None:
        active = [f for f in self.active_faults if f.active]
        self.highest_fault_level = max((f.level for f in active), default=FaultLevel.NONE)
        self.last_fault = active[-1].fault_type if active else FaultType.NONE


# =========================================================
# Mission bookkeeping / RL records
# =========================================================

@dataclass
class RewardBreakdown:
    """
    Keeps reward accounting interpretable and debuggable.
    """
    total: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    events: List[RewardEvent] = field(default_factory=list)

    def add(self, key: str, value: float, event: Optional[RewardEvent] = None) -> None:
        self.total += value
        self.components[key] = self.components.get(key, 0.0) + value
        if event is not None:
            self.events.append(event)

    def reset(self) -> None:
        self.total = 0.0
        self.components.clear()
        self.events.clear()


@dataclass
class EpisodeStats:
    """
    Accumulated mission-level statistics.
    """
    total_reward: float = 0.0
    useful_images_captured: int = 0
    cloudy_images_captured: int = 0
    useful_images_discarded: int = 0
    useful_images_stored: int = 0
    useful_images_downlinked: int = 0

    total_data_downlinked_mb: float = 0.0
    total_energy_generated_wh: float = 0.0
    total_energy_consumed_wh: float = 0.0

    safe_mode_entries: int = 0
    fault_count: int = 0
    downlink_passes_used: int = 0

    def update_reward(self, reward: float) -> None:
        self.total_reward += reward


@dataclass
class ActionRecord:
    """
    Useful for logging, replay, and debugging policies.
    """
    step_count: int
    sim_time_s: float
    action: Action
    valid: bool = True
    reason: str = ""


@dataclass
class StepInfo:
    """
    Info dict payload replacement with strong typing.
    """
    action_taken: Action = Action.NO_OP
    reward_breakdown: RewardBreakdown = field(default_factory=RewardBreakdown)
    done: bool = False
    end_reason: EpisodeEndReason = EpisodeEndReason.NOT_DONE
    notes: List[str] = field(default_factory=list)

    def add_note(self, msg: str) -> None:
        self.notes.append(msg)


# =========================================================
# Main spacecraft state
# =========================================================

@dataclass
class SpacecraftState:
    """
    Full simulator state.

    This is the internal continuous/hybrid state of the spacecraft,
    not necessarily the same as the RL observation.
    """
    mode: SpacecraftMode = SpacecraftMode.BOOT

    time: SimTime = field(default_factory=SimTime)
    orbit: OrbitState = field(default_factory=OrbitState)

    eps: EPSState = field(default_factory=EPSState)
    thermal: ThermalState = field(default_factory=ThermalState)
    adcs: ADCSState = field(default_factory=ADCSState)
    payload: PayloadState = field(default_factory=PayloadState)
    comms: CommsState = field(default_factory=CommsState)
    cdh: CDHState = field(default_factory=CDHState)
    faults: FaultState = field(default_factory=FaultState)

    last_action: Action = Action.NO_OP
    previous_action: Action = Action.NO_OP

    alive: bool = True
    done: bool = False
    end_reason: EpisodeEndReason = EpisodeEndReason.NOT_DONE

    step_reward: RewardBreakdown = field(default_factory=RewardBreakdown)
    episode_stats: EpisodeStats = field(default_factory=EpisodeStats)

    action_history: List[ActionRecord] = field(default_factory=list)

    def begin_step(self, action: Action) -> None:
        self.previous_action = self.last_action
        self.last_action = action
        self.step_reward.reset()
        self.action_history.append(
            ActionRecord(
                step_count=self.time.step_count,
                sim_time_s=self.time.sim_time_s,
                action=action,
            )
        )

    def end_step(self, reward: float) -> None:
        self.episode_stats.update_reward(reward)

    def terminate(self, reason: EpisodeEndReason) -> None:
        self.alive = False
        self.done = True
        self.end_reason = reason

    @property
    def power_positive(self) -> bool:
        return self.eps.net_power_w >= 0.0

    @property
    def link_available(self) -> bool:
        return self.comms.link_available

    @property
    def memory_pressure(self) -> MemoryPressure:
        return self.cdh.memory_pressure

    @property
    def battery_band(self) -> BatteryBand:
        return self.eps.battery_band

    @property
    def battery_temp_band(self) -> TemperatureBand:
        return self.thermal.battery_temp_band

    @property
    def payload_temp_band(self) -> TemperatureBand:
        return self.thermal.payload_temp_band

    @property
    def pointing_quality(self) -> PointingQuality:
        return self.adcs.pointing_quality

    @property
    def critical_fault(self) -> bool:
        return self.faults.has_level_at_least(FaultLevel.CRITICAL)

    @property
    def storage_full(self) -> bool:
        return self.cdh.memory_free_mb <= 0.0

    @property
    def observation_dict(self) -> Dict[str, Any]:
        """
        A compact default observation dictionary.
        The observation_builder can produce richer/binned forms later.
        """
        return {
            "mode": int(self.mode),
            "orbit_phase": self.orbit.orbit_phase,
            "sunlit": int(self.orbit.sunlight_state),
            "sun_elevation_deg": self.orbit.sun_elevation_deg,
            "over_target": int(self.orbit.over_target),
            "target_opportunity": int(self.orbit.target_opportunity),
            "battery_soc_pct": self.eps.battery_soc_pct,
            "battery_band": int(self.eps.battery_band),
            "battery_temp_c": self.thermal.battery_temp_c,
            "battery_temp_band": int(self.thermal.battery_temp_band),
            "solar_input_w": self.eps.solar_input_w,
            "load_output_w": self.eps.load_output_w,
            "net_power_w": self.eps.net_power_w,
            "attitude_mode": int(self.adcs.mode),
            "pointing_error_deg": self.adcs.pointing_error_deg,
            "pointing_quality": int(self.adcs.pointing_quality),
            "wheel_momentum_nms": self.adcs.wheel_momentum_nms,
            "payload_mode": int(self.payload.mode),
            "has_frame": int(self.payload.has_frame),
            "frame_class": int(self.payload.current_frame_class),
            "frame_cloud_prob": self.payload.current_frame_cloud_prob,
            "frame_usefulness": self.payload.current_frame_usefulness,
            "classifier_confidence": self.payload.classifier_confidence,
            "comms_mode": int(self.comms.mode),
            "gs_visible": int(self.comms.gs_visible),
            "pass_state": int(self.comms.pass_state),
            "link_quality": int(self.comms.link_quality),
            "gs_elevation_deg": self.comms.gs_elevation_deg,
            "memory_used_mb": self.cdh.memory_used_mb,
            "memory_free_mb": self.cdh.memory_free_mb,
            "memory_pressure": int(self.cdh.memory_pressure),
            "downlink_queue_mb": self.cdh.downlink_queue_mb,
            "fault_level": int(self.faults.highest_fault_level),
            "last_fault": int(self.faults.last_fault),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# Optional RL-facing observation structure
# =========================================================

@dataclass
class AgentObservation:
    """
    Structured RL observation.

    Keep this separate from SpacecraftState so the simulator can remain
    physically richer than what the agent directly sees.
    """
    orbit_phase_norm: float
    sunlit: int
    target_available: int
    target_opportunity: int

    battery_soc_norm: float
    battery_band: int
    battery_temp_band: int
    power_positive: int

    memory_utilization_norm: float
    memory_pressure: int
    queue_norm: float

    attitude_mode: int
    pointing_quality: int
    wheels_saturated: int

    payload_mode: int
    has_frame: int
    frame_class: int
    frame_usefulness_norm: float
    classifier_confidence_norm: float

    gs_visible: int
    pass_state: int
    link_quality: int

    safe_mode_latched: int
    highest_fault_level: int

    def as_vector(self) -> List[float]:
        return [
            self.orbit_phase_norm,
            float(self.sunlit),
            float(self.target_available),
            float(self.target_opportunity),
            self.battery_soc_norm,
            float(self.battery_band),
            float(self.battery_temp_band),
            float(self.power_positive),
            self.memory_utilization_norm,
            float(self.memory_pressure),
            self.queue_norm,
            float(self.attitude_mode),
            float(self.pointing_quality),
            float(self.wheels_saturated),
            float(self.payload_mode),
            float(self.has_frame),
            float(self.frame_class),
            self.frame_usefulness_norm,
            self.classifier_confidence_norm,
            float(self.gs_visible),
            float(self.pass_state),
            float(self.link_quality),
            float(self.safe_mode_latched),
            float(self.highest_fault_level),
        ]

    @classmethod
    def from_spacecraft_state(cls, state: SpacecraftState) -> "AgentObservation":
        queue_norm = normalize(state.cdh.downlink_queue_mb, 0.0, state.cdh.memory_capacity_mb)
        return cls(
            orbit_phase_norm=state.orbit.orbit_phase,
            sunlit=int(state.orbit.sunlight_state),
            target_available=int(state.orbit.over_target),
            target_opportunity=int(state.orbit.target_opportunity),
            battery_soc_norm=normalize(state.eps.battery_soc_pct, 0.0, 100.0),
            battery_band=int(state.eps.battery_band),
            battery_temp_band=int(state.thermal.battery_temp_band),
            power_positive=int(state.power_positive),
            memory_utilization_norm=state.cdh.memory_utilization,
            memory_pressure=int(state.cdh.memory_pressure),
            queue_norm=queue_norm,
            attitude_mode=int(state.adcs.mode),
            pointing_quality=int(state.adcs.pointing_quality),
            wheels_saturated=int(state.adcs.wheels_saturated),
            payload_mode=int(state.payload.mode),
            has_frame=int(state.payload.has_frame),
            frame_class=int(state.payload.current_frame_class),
            frame_usefulness_norm=clamp(state.payload.current_frame_usefulness, 0.0, 1.0),
            classifier_confidence_norm=clamp(state.payload.classifier_confidence, 0.0, 1.0),
            gs_visible=int(state.comms.gs_visible),
            pass_state=int(state.comms.pass_state),
            link_quality=int(state.comms.link_quality),
            safe_mode_latched=int(state.faults.safe_mode_latched),
            highest_fault_level=int(state.faults.highest_fault_level),
        )


# =========================================================
# Factory
# =========================================================

def create_default_spacecraft_state() -> SpacecraftState:
    """
    Standard simulator boot state.
    """
    state = SpacecraftState()

    state.mode = SpacecraftMode.BOOT

    state.eps.update_net_power()

    state.orbit.sunlight_state = SunlightState.SUNLIT
    state.comms.pass_state = GroundPassState.NONE
    state.comms.link_quality = LinkQuality.NONE

    state.payload.mode = PayloadMode.OFF
    state.cdh.mode = DataHandlingMode.IDLE
    state.thermal.mode = ThermalMode.NOMINAL
    state.eps.mode = PowerMode.NOMINAL
    state.adcs.mode = AttitudeMode.SAFE_SUN_ACQUIRE

    return state