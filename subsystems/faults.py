from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models.enums import (
    Action,
    FaultLevel,
    FaultType,
    SpacecraftMode,
    SubsystemName,
)
from models.state_models import SpacecraftState


# =========================================================
# Configuration
# =========================================================

@dataclass(frozen=True)
class FaultConfig:
    """
    Fault Detection, Isolation, and Recovery (FDIR) configuration.

    This module is designed for an operational CubeSat simulator, not a purely
    academic toy model. It detects subsystem health limit violations, escalates
    fault severity, latches safe mode when needed, and supports controlled
    recovery/clear logic.

    Philosophy:
    - warning faults = watch closely
    - limit faults = subsystem seriously degraded
    - critical faults = spacecraft mission at risk
    - fatal faults = end-of-mission / terminal

    This layer should remain deterministic unless the environment explicitly
    chooses to inject random faults elsewhere.
    """

    # -----------------------------------------------------
    # EPS / battery
    # -----------------------------------------------------
    low_battery_warning_soc_pct: float = 25.0
    low_battery_limit_soc_pct: float = 15.0
    low_battery_critical_soc_pct: float = 7.5
    low_battery_fatal_soc_pct: float = 1.0

    brownout_warning_enabled: bool = True

    # -----------------------------------------------------
    # Battery temperature
    # -----------------------------------------------------
    battery_cold_warning_c: float = 2.0
    battery_cold_limit_c: float = 0.0
    battery_cold_critical_c: float = -3.0

    battery_hot_warning_c: float = 33.0
    battery_hot_limit_c: float = 37.0
    battery_hot_critical_c: float = 42.0

    # -----------------------------------------------------
    # Bus / payload thermal
    # -----------------------------------------------------
    payload_hot_warning_c: float = 35.0
    payload_hot_limit_c: float = 40.0
    payload_hot_critical_c: float = 45.0

    payload_cold_warning_c: float = -2.0
    payload_cold_limit_c: float = -5.0
    payload_cold_critical_c: float = -8.0

    bus_hot_limit_c: float = 50.0
    bus_hot_critical_c: float = 55.0
    bus_cold_limit_c: float = -10.0
    bus_cold_critical_c: float = -15.0

    # -----------------------------------------------------
    # ADCS
    # -----------------------------------------------------
    adcs_pointing_lost_warning_deg: float = 4.0
    adcs_pointing_lost_limit_deg: float = 8.0
    adcs_pointing_lost_critical_deg: float = 14.0

    wheel_saturation_warning_frac: float = 0.90
    wheel_saturation_limit_frac: float = 1.00
    wheel_saturation_critical_frac: float = 1.15

    body_rate_limit_warning_dps: float = 2.5
    body_rate_limit_critical_dps: float = 6.0

    # -----------------------------------------------------
    # Payload / classifier
    # -----------------------------------------------------
    classifier_failure_if_processing_without_frame: bool = True
    payload_stuck_max_warmup_steps: int = 40

    # -----------------------------------------------------
    # Comms
    # -----------------------------------------------------
    comms_timeout_if_tx_without_visibility_steps: int = 5
    comms_link_loss_if_tx_rate_zero_steps: int = 5
    downlink_abort_if_queue_not_reducing_steps: int = 8

    # -----------------------------------------------------
    # CDH / storage
    # -----------------------------------------------------
    memory_full_warning_frac: float = 0.90
    memory_full_limit_frac: float = 0.97
    memory_full_critical_frac: float = 1.02

    cdh_reset_loop_limit_count: int = 3
    cdh_reset_loop_critical_count: int = 6

    # -----------------------------------------------------
    # Solar generation / power anomalies
    # -----------------------------------------------------
    solar_generation_loss_if_sunlit_and_low_input_w: float = 0.25

    # -----------------------------------------------------
    # Safe mode latch policy
    # -----------------------------------------------------
    latch_safe_mode_on_limit_faults: Tuple[FaultType, ...] = (
        FaultType.BATTERY_OVERTEMP,
        FaultType.BATTERY_UNDERTEMP,
        FaultType.PAYLOAD_OVERTEMP,
        FaultType.COMMS_TIMEOUT,
        FaultType.COMMS_LINK_LOSS,
        FaultType.STORAGE_CORRUPTION,
        FaultType.CDH_RESET_LOOP,
    )

    latch_safe_mode_on_critical_any: bool = True
    latch_safe_mode_on_fatal_any: bool = True

    # -----------------------------------------------------
    # Fault recovery behavior
    # -----------------------------------------------------
    clear_warning_faults_on_nominal_recovery: bool = True
    clear_limit_faults_on_recovery_action: bool = True
    clear_comms_faults_on_reset_comms: bool = True
    clear_payload_faults_on_reset_payload: bool = True
    clear_adcs_faults_on_reset_adcs: bool = True
    clear_cdh_faults_on_reset_cdh: bool = True


DEFAULT_FAULT_CONFIG = FaultConfig()


# =========================================================
# Internal counters
# =========================================================

@dataclass
class FaultCounters:
    """
    Persistent counters used to detect faults that depend on duration or repeated
    failed behavior rather than one-step threshold crossings.
    """
    payload_warmup_steps: int = 0

    comms_tx_without_visibility_steps: int = 0
    comms_tx_rate_zero_steps: int = 0
    comms_queue_not_reducing_steps: int = 0

    previous_downlink_queue_mb: float = 0.0

    def reset(self) -> None:
        self.payload_warmup_steps = 0
        self.comms_tx_without_visibility_steps = 0
        self.comms_tx_rate_zero_steps = 0
        self.comms_queue_not_reducing_steps = 0
        self.previous_downlink_queue_mb = 0.0


# =========================================================
# Breakdown
# =========================================================

@dataclass(frozen=True)
class FaultBreakdown:
    """
    Detailed fault update report.
    """
    highest_fault_level: FaultLevel
    active_fault_count: int
    safe_mode_latched: bool
    last_fault: FaultType
    newly_added_faults: int
    cleared_faults: int


# =========================================================
# Fault subsystem
# =========================================================

class FaultSubsystem:
    """
    FDIR subsystem for the CubeSat simulator.

    Responsibilities:
    - detect faults from spacecraft state
    - escalate or clear active faults
    - maintain safe-mode latch logic
    - respond to reset / recovery actions
    """

    def __init__(self, config: FaultConfig = DEFAULT_FAULT_CONFIG):
        self.config = config
        self.counters = FaultCounters()

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def initialize(self, state: SpacecraftState) -> None:
        self.counters.reset()
        self.counters.previous_downlink_queue_mb = state.cdh.downlink_queue_mb

    def step(
        self,
        state: SpacecraftState,
        dt_s: float,
        action: Optional[Action] = None,
    ) -> FaultBreakdown:
        """
        Evaluate subsystem state, update active faults, and apply safe-mode latching.
        """
        before_active = self._active_fault_keys(state)

        # Recovery / reset actions first, so current step can clear old faults
        self._apply_recovery_actions(state, action)

        # Update duration-based counters
        self._update_counters(state)

        # Run fault detectors
        self._detect_eps_faults(state)
        self._detect_thermal_faults(state)
        self._detect_adcs_faults(state)
        self._detect_payload_faults(state)
        self._detect_comms_faults(state)
        self._detect_cdh_faults(state)

        # Clear automatically recoverable faults if state is nominal
        self._clear_recovered_faults(state)

        # Update safe mode latch
        self._update_safe_mode_latch(state)

        after_active = self._active_fault_keys(state)
        newly_added = len(after_active - before_active)
        cleared = len(before_active - after_active)

        self.counters.previous_downlink_queue_mb = state.cdh.downlink_queue_mb

        return FaultBreakdown(
            highest_fault_level=state.faults.highest_fault_level,
            active_fault_count=len([f for f in state.faults.active_faults if f.active]),
            safe_mode_latched=state.faults.safe_mode_latched,
            last_fault=state.faults.last_fault,
            newly_added_faults=newly_added,
            cleared_faults=cleared,
        )

    # -----------------------------------------------------
    # Fault detectors
    # -----------------------------------------------------

    def _detect_eps_faults(self, state: SpacecraftState) -> None:
        soc = state.eps.battery_soc_pct

        # LOW_BATTERY
        if soc <= self.config.low_battery_fatal_soc_pct:
            self._set_fault(
                state,
                SubsystemName.EPS,
                FaultType.LOW_BATTERY,
                FaultLevel.FATAL,
                f"Battery SoC critically depleted at {soc:.2f}%.",
            )
        elif soc <= self.config.low_battery_critical_soc_pct:
            self._set_fault(
                state,
                SubsystemName.EPS,
                FaultType.LOW_BATTERY,
                FaultLevel.CRITICAL,
                f"Battery SoC critically low at {soc:.2f}%.",
            )
        elif soc <= self.config.low_battery_limit_soc_pct:
            self._set_fault(
                state,
                SubsystemName.EPS,
                FaultType.LOW_BATTERY,
                FaultLevel.LIMIT,
                f"Battery SoC below limit at {soc:.2f}%.",
            )
        elif soc <= self.config.low_battery_warning_soc_pct:
            self._set_fault(
                state,
                SubsystemName.EPS,
                FaultType.LOW_BATTERY,
                FaultLevel.WARNING,
                f"Battery SoC low at {soc:.2f}%.",
            )
        else:
            self._clear_fault(state, SubsystemName.EPS, FaultType.LOW_BATTERY)

        # Solar generation loss
        if (
            state.orbit.is_sunlit
            and state.eps.solar_input_w <= self.config.solar_generation_loss_if_sunlit_and_low_input_w
            and state.eps.mode.name not in ("CRITICAL_LOW_POWER",)
        ):
            self._set_fault(
                state,
                SubsystemName.EPS,
                FaultType.SOLAR_GENERATION_LOSS,
                FaultLevel.WARNING,
                f"Very low solar generation while sunlit: {state.eps.solar_input_w:.2f} W.",
            )
        else:
            self._clear_fault(state, SubsystemName.EPS, FaultType.SOLAR_GENERATION_LOSS)

    def _detect_thermal_faults(self, state: SpacecraftState) -> None:
        bt = state.thermal.battery_temp_c

        # Battery under-temp
        if bt <= self.config.battery_cold_critical_c:
            self._set_fault(
                state,
                SubsystemName.THERMAL,
                FaultType.BATTERY_UNDERTEMP,
                FaultLevel.CRITICAL,
                f"Battery temperature too cold: {bt:.2f} C.",
            )
        elif bt <= self.config.battery_cold_limit_c:
            self._set_fault(
                state,
                SubsystemName.THERMAL,
                FaultType.BATTERY_UNDERTEMP,
                FaultLevel.LIMIT,
                f"Battery temperature below limit: {bt:.2f} C.",
            )
        elif bt <= self.config.battery_cold_warning_c:
            self._set_fault(
                state,
                SubsystemName.THERMAL,
                FaultType.BATTERY_UNDERTEMP,
                FaultLevel.WARNING,
                f"Battery temperature low: {bt:.2f} C.",
            )
        else:
            self._clear_fault(state, SubsystemName.THERMAL, FaultType.BATTERY_UNDERTEMP)

        # Battery over-temp
        if bt >= self.config.battery_hot_critical_c:
            self._set_fault(
                state,
                SubsystemName.THERMAL,
                FaultType.BATTERY_OVERTEMP,
                FaultLevel.CRITICAL,
                f"Battery temperature too hot: {bt:.2f} C.",
            )
        elif bt >= self.config.battery_hot_limit_c:
            self._set_fault(
                state,
                SubsystemName.THERMAL,
                FaultType.BATTERY_OVERTEMP,
                FaultLevel.LIMIT,
                f"Battery temperature above limit: {bt:.2f} C.",
            )
        elif bt >= self.config.battery_hot_warning_c:
            self._set_fault(
                state,
                SubsystemName.THERMAL,
                FaultType.BATTERY_OVERTEMP,
                FaultLevel.WARNING,
                f"Battery temperature warm: {bt:.2f} C.",
            )
        else:
            self._clear_fault(state, SubsystemName.THERMAL, FaultType.BATTERY_OVERTEMP)

        # Payload over-temp
        pt = state.thermal.payload_temp_c
        if pt >= self.config.payload_hot_critical_c:
            self._set_fault(
                state,
                SubsystemName.PAYLOAD,
                FaultType.PAYLOAD_OVERTEMP,
                FaultLevel.CRITICAL,
                f"Payload temperature too hot: {pt:.2f} C.",
            )
        elif pt >= self.config.payload_hot_limit_c:
            self._set_fault(
                state,
                SubsystemName.PAYLOAD,
                FaultType.PAYLOAD_OVERTEMP,
                FaultLevel.LIMIT,
                f"Payload temperature above limit: {pt:.2f} C.",
            )
        elif pt >= self.config.payload_hot_warning_c:
            self._set_fault(
                state,
                SubsystemName.PAYLOAD,
                FaultType.PAYLOAD_OVERTEMP,
                FaultLevel.WARNING,
                f"Payload temperature warm: {pt:.2f} C.",
            )
        else:
            self._clear_fault(state, SubsystemName.PAYLOAD, FaultType.PAYLOAD_OVERTEMP)

        # Generic thermal limits hot/cold from bus
        bus = state.thermal.bus_temp_c
        if bus >= self.config.bus_hot_critical_c or bus <= self.config.bus_cold_critical_c:
            self._set_fault(
                state,
                SubsystemName.THERMAL,
                FaultType.THERMAL_LIMIT_HOT if bus >= self.config.bus_hot_critical_c else FaultType.THERMAL_LIMIT_COLD,
                FaultLevel.CRITICAL,
                f"Bus thermal critical limit reached: {bus:.2f} C.",
            )
        elif bus >= self.config.bus_hot_limit_c or bus <= self.config.bus_cold_limit_c:
            self._set_fault(
                state,
                SubsystemName.THERMAL,
                FaultType.THERMAL_LIMIT_HOT if bus >= self.config.bus_hot_limit_c else FaultType.THERMAL_LIMIT_COLD,
                FaultLevel.LIMIT,
                f"Bus thermal limit reached: {bus:.2f} C.",
            )
        else:
            self._clear_fault(state, SubsystemName.THERMAL, FaultType.THERMAL_LIMIT_HOT)
            self._clear_fault(state, SubsystemName.THERMAL, FaultType.THERMAL_LIMIT_COLD)

    def _detect_adcs_faults(self, state: SpacecraftState) -> None:
        pe = state.adcs.pointing_error_deg
        if pe >= self.config.adcs_pointing_lost_critical_deg:
            self._set_fault(
                state,
                SubsystemName.ADCS,
                FaultType.ADCS_POINTING_LOST,
                FaultLevel.CRITICAL,
                f"Pointing error critical at {pe:.2f} deg.",
            )
        elif pe >= self.config.adcs_pointing_lost_limit_deg:
            self._set_fault(
                state,
                SubsystemName.ADCS,
                FaultType.ADCS_POINTING_LOST,
                FaultLevel.LIMIT,
                f"Pointing error above limit at {pe:.2f} deg.",
            )
        elif pe >= self.config.adcs_pointing_lost_warning_deg:
            self._set_fault(
                state,
                SubsystemName.ADCS,
                FaultType.ADCS_POINTING_LOST,
                FaultLevel.WARNING,
                f"Pointing error elevated at {pe:.2f} deg.",
            )
        else:
            self._clear_fault(state, SubsystemName.ADCS, FaultType.ADCS_POINTING_LOST)

        # Wheel saturation
        util = state.adcs.wheel_momentum_nms / max(state.adcs.wheel_momentum_limit_nms, 1e-9)
        if util >= self.config.wheel_saturation_critical_frac:
            self._set_fault(
                state,
                SubsystemName.ADCS,
                FaultType.WHEEL_SATURATION,
                FaultLevel.CRITICAL,
                f"Wheel momentum critical utilization {util:.2f}.",
            )
        elif util >= self.config.wheel_saturation_limit_frac:
            self._set_fault(
                state,
                SubsystemName.ADCS,
                FaultType.WHEEL_SATURATION,
                FaultLevel.LIMIT,
                f"Wheel momentum at/above saturation {util:.2f}.",
            )
        elif util >= self.config.wheel_saturation_warning_frac:
            self._set_fault(
                state,
                SubsystemName.ADCS,
                FaultType.WHEEL_SATURATION,
                FaultLevel.WARNING,
                f"Wheel momentum high utilization {util:.2f}.",
            )
        else:
            self._clear_fault(state, SubsystemName.ADCS, FaultType.WHEEL_SATURATION)

        # Sensor degraded
        if not (state.adcs.imu_available and state.adcs.magnetometer_available and state.adcs.star_tracker_available):
            self._set_fault(
                state,
                SubsystemName.ADCS,
                FaultType.SENSOR_DEGRADED,
                FaultLevel.WARNING,
                "One or more primary ADCS sensors unavailable.",
            )
        else:
            self._clear_fault(state, SubsystemName.ADCS, FaultType.SENSOR_DEGRADED)

    def _detect_payload_faults(self, state: SpacecraftState) -> None:
        # Payload stuck in warmup too long
        if state.payload.mode.name == "WARMUP":
            if self.counters.payload_warmup_steps >= self.config.payload_stuck_max_warmup_steps:
                self._set_fault(
                    state,
                    SubsystemName.PAYLOAD,
                    FaultType.PAYLOAD_STUCK,
                    FaultLevel.LIMIT,
                    f"Payload stuck in warmup for {self.counters.payload_warmup_steps} steps.",
                )
        else:
            self._clear_fault(state, SubsystemName.PAYLOAD, FaultType.PAYLOAD_STUCK)

        # Classifier failure proxy
        if (
            self.config.classifier_failure_if_processing_without_frame
            and state.payload.mode.name == "PROCESSING"
            and not state.payload.has_frame
        ):
            self._set_fault(
                state,
                SubsystemName.PAYLOAD,
                FaultType.CLASSIFIER_FAILURE,
                FaultLevel.WARNING,
                "Payload processing active without an available frame.",
            )
        else:
            self._clear_fault(state, SubsystemName.PAYLOAD, FaultType.CLASSIFIER_FAILURE)

    def _detect_comms_faults(self, state: SpacecraftState) -> None:
        # TX without visibility -> timeout
        if self.counters.comms_tx_without_visibility_steps >= self.config.comms_timeout_if_tx_without_visibility_steps:
            self._set_fault(
                state,
                SubsystemName.COMMS,
                FaultType.COMMS_TIMEOUT,
                FaultLevel.LIMIT,
                f"TX attempted without visibility for {self.counters.comms_tx_without_visibility_steps} steps.",
            )
        else:
            self._clear_fault(state, SubsystemName.COMMS, FaultType.COMMS_TIMEOUT)

        # TX active but zero effective rate -> link loss
        if self.counters.comms_tx_rate_zero_steps >= self.config.comms_link_loss_if_tx_rate_zero_steps:
            self._set_fault(
                state,
                SubsystemName.COMMS,
                FaultType.COMMS_LINK_LOSS,
                FaultLevel.LIMIT,
                f"TX active with zero rate for {self.counters.comms_tx_rate_zero_steps} steps.",
            )
        else:
            self._clear_fault(state, SubsystemName.COMMS, FaultType.COMMS_LINK_LOSS)

        # Queue not reducing while downlinking -> aborted/failed downlink
        if self.counters.comms_queue_not_reducing_steps >= self.config.downlink_abort_if_queue_not_reducing_steps:
            self._set_fault(
                state,
                SubsystemName.COMMS,
                FaultType.DOWNLINK_ABORTED,
                FaultLevel.WARNING,
                f"Downlink queue not reducing for {self.counters.comms_queue_not_reducing_steps} steps.",
            )
        else:
            self._clear_fault(state, SubsystemName.COMMS, FaultType.DOWNLINK_ABORTED)

    def _detect_cdh_faults(self, state: SpacecraftState) -> None:
        util = state.cdh.memory_used_mb / max(state.cdh.memory_capacity_mb, 1e-9)

        if util >= self.config.memory_full_critical_frac:
            self._set_fault(
                state,
                SubsystemName.CDH,
                FaultType.MEMORY_FULL,
                FaultLevel.CRITICAL,
                f"Memory utilization critical at {util:.3f}.",
            )
        elif util >= self.config.memory_full_limit_frac:
            self._set_fault(
                state,
                SubsystemName.CDH,
                FaultType.MEMORY_FULL,
                FaultLevel.LIMIT,
                f"Memory utilization above limit at {util:.3f}.",
            )
        elif util >= self.config.memory_full_warning_frac:
            self._set_fault(
                state,
                SubsystemName.CDH,
                FaultType.MEMORY_FULL,
                FaultLevel.WARNING,
                f"Memory utilization high at {util:.3f}.",
            )
        else:
            self._clear_fault(state, SubsystemName.CDH, FaultType.MEMORY_FULL)

        if state.cdh.storage_corrupted:
            self._set_fault(
                state,
                SubsystemName.CDH,
                FaultType.STORAGE_CORRUPTION,
                FaultLevel.CRITICAL,
                "Storage corruption flag active.",
            )
        else:
            self._clear_fault(state, SubsystemName.CDH, FaultType.STORAGE_CORRUPTION)

        if state.cdh.reset_count >= self.config.cdh_reset_loop_critical_count:
            self._set_fault(
                state,
                SubsystemName.CDH,
                FaultType.CDH_RESET_LOOP,
                FaultLevel.CRITICAL,
                f"CDH reset loop critical count reached: {state.cdh.reset_count}.",
            )
        elif state.cdh.reset_count >= self.config.cdh_reset_loop_limit_count:
            self._set_fault(
                state,
                SubsystemName.CDH,
                FaultType.CDH_RESET_LOOP,
                FaultLevel.LIMIT,
                f"CDH reset loop limit count reached: {state.cdh.reset_count}.",
            )
        else:
            self._clear_fault(state, SubsystemName.CDH, FaultType.CDH_RESET_LOOP)

    # -----------------------------------------------------
    # Counter update
    # -----------------------------------------------------

    def _update_counters(self, state: SpacecraftState) -> None:
        # Payload warmup duration
        if state.payload.mode.name == "WARMUP":
            self.counters.payload_warmup_steps += 1
        else:
            self.counters.payload_warmup_steps = 0

        # Comms duration-based anomalies
        tx_mode = state.comms.tx_enabled

        if tx_mode and not state.comms.gs_visible:
            self.counters.comms_tx_without_visibility_steps += 1
        else:
            self.counters.comms_tx_without_visibility_steps = 0

        if tx_mode and state.comms.gs_visible and state.comms.current_rate_mbps <= 0.0:
            self.counters.comms_tx_rate_zero_steps += 1
        else:
            self.counters.comms_tx_rate_zero_steps = 0

        if tx_mode and state.comms.gs_visible and state.cdh.downlink_queue_mb > 0.0:
            if state.cdh.downlink_queue_mb >= self.counters.previous_downlink_queue_mb - 1e-9:
                self.counters.comms_queue_not_reducing_steps += 1
            else:
                self.counters.comms_queue_not_reducing_steps = 0
        else:
            self.counters.comms_queue_not_reducing_steps = 0

    # -----------------------------------------------------
    # Recovery / reset actions
    # -----------------------------------------------------

    def _apply_recovery_actions(self, state: SpacecraftState, action: Optional[Action]) -> None:
        if action is None:
            return

        if action == Action.FAULT_RECOVERY:
            # Clear warnings broadly if system is nominal enough
            if self.config.clear_warning_faults_on_nominal_recovery:
                self._clear_faults_by_level(state, FaultLevel.WARNING)

            if self.config.clear_limit_faults_on_recovery_action:
                for ft in (
                    FaultType.DOWNLINK_ABORTED,
                    FaultType.COMMS_TIMEOUT,
                    FaultType.COMMS_LINK_LOSS,
                    FaultType.PAYLOAD_STUCK,
                ):
                    self._clear_fault_any_subsystem(state, ft)

        elif action == Action.RESET_COMMS and self.config.clear_comms_faults_on_reset_comms:
            for ft in (
                FaultType.COMMS_TIMEOUT,
                FaultType.COMMS_LINK_LOSS,
                FaultType.DOWNLINK_ABORTED,
            ):
                self._clear_fault_any_subsystem(state, ft)

        elif action == Action.RESET_PAYLOAD and self.config.clear_payload_faults_on_reset_payload:
            for ft in (
                FaultType.PAYLOAD_STUCK,
                FaultType.PAYLOAD_OVERTEMP,
                FaultType.CLASSIFIER_FAILURE,
            ):
                self._clear_fault_any_subsystem(state, ft)

        elif action == Action.RESET_ADCS and self.config.clear_adcs_faults_on_reset_adcs:
            for ft in (
                FaultType.ADCS_POINTING_LOST,
                FaultType.WHEEL_SATURATION,
                FaultType.SENSOR_DEGRADED,
            ):
                self._clear_fault_any_subsystem(state, ft)

        elif action == Action.RESET_CDH and self.config.clear_cdh_faults_on_reset_cdh:
            for ft in (
                FaultType.MEMORY_FULL,
                FaultType.STORAGE_CORRUPTION,
                FaultType.CDH_RESET_LOOP,
            ):
                self._clear_fault_any_subsystem(state, ft)

    # -----------------------------------------------------
    # Auto-clear logic
    # -----------------------------------------------------

    def _clear_recovered_faults(self, state: SpacecraftState) -> None:
        """
        Automatically clear certain warning faults when their triggering conditions
        are no longer present.
        """
        # COMMS link-loss-like warnings are handled in detectors already.
        # LOW_BATTERY clears automatically via threshold detector.
        # WHEEL_SATURATION clears automatically via threshold detector.
        # STORAGE_CORRUPTION generally does not auto-clear unless reset/recovery did it.
        pass

    # -----------------------------------------------------
    # Safe mode latch logic
    # -----------------------------------------------------

    def _update_safe_mode_latch(self, state: SpacecraftState) -> None:
        active = [f for f in state.faults.active_faults if f.active]

        should_latch = False

        if self.config.latch_safe_mode_on_fatal_any and any(f.level == FaultLevel.FATAL for f in active):
            should_latch = True

        if self.config.latch_safe_mode_on_critical_any and any(f.level == FaultLevel.CRITICAL for f in active):
            should_latch = True

        if any(
            f.fault_type in self.config.latch_safe_mode_on_limit_faults and f.level >= FaultLevel.LIMIT
            for f in active
        ):
            should_latch = True

        state.faults.safe_mode_latched = should_latch

        if should_latch and state.mode not in (
            SpacecraftMode.SAFE,
            SpacecraftMode.SURVIVAL,
            SpacecraftMode.FAULT_RECOVERY,
        ):
            state.mode = SpacecraftMode.SAFE

    # -----------------------------------------------------
    # Fault manipulation helpers
    # -----------------------------------------------------

    def _set_fault(
        self,
        state: SpacecraftState,
        subsystem: SubsystemName,
        fault_type: FaultType,
        level: FaultLevel,
        message: str,
    ) -> None:
        state.faults.add_fault(
            subsystem=subsystem,
            fault_type=fault_type,
            level=level,
            step_count=state.time.step_count,
            message=message,
        )

    def _clear_fault(
        self,
        state: SpacecraftState,
        subsystem: SubsystemName,
        fault_type: FaultType,
    ) -> None:
        state.faults.clear_fault(subsystem, fault_type)

    def _clear_fault_any_subsystem(
        self,
        state: SpacecraftState,
        fault_type: FaultType,
    ) -> None:
        for subsystem in (
            SubsystemName.EPS,
            SubsystemName.ADCS,
            SubsystemName.PAYLOAD,
            SubsystemName.COMMS,
            SubsystemName.CDH,
            SubsystemName.THERMAL,
        ):
            state.faults.clear_fault(subsystem, fault_type)

    def _clear_faults_by_level(
        self,
        state: SpacecraftState,
        level: FaultLevel,
    ) -> None:
        for fault in state.faults.active_faults:
            if fault.active and fault.level == level:
                fault.active = False
        state.faults._recompute()

    def _active_fault_keys(self, state: SpacecraftState) -> set[tuple[SubsystemName, FaultType, FaultLevel]]:
        return {
            (f.subsystem, f.fault_type, f.level)
            for f in state.faults.active_faults
            if f.active
        }


# =========================================================
# Functional helpers
# =========================================================

def initialize_fault_state(
    state: SpacecraftState,
    config: FaultConfig = DEFAULT_FAULT_CONFIG,
) -> FaultSubsystem:
    subsystem = FaultSubsystem(config=config)
    subsystem.initialize(state)
    return subsystem


def update_fault_state(
    state: SpacecraftState,
    subsystem: FaultSubsystem,
    dt_s: float,
    action: Optional[Action] = None,
) -> FaultBreakdown:
    return subsystem.step(
        state=state,
        dt_s=dt_s,
        action=action,
    )


# =========================================================
# Smoke test helper
# =========================================================

def faults_smoke_summary(
    state: SpacecraftState,
    config: FaultConfig = DEFAULT_FAULT_CONFIG,
) -> Dict[str, float]:
    """
    Quick FDIR sanity-check helper.
    """
    subsystem = FaultSubsystem(config=config)
    subsystem.initialize(state)

    # Force a few typical fault conditions
    state.eps.battery_soc_pct = 6.0
    state.thermal.battery_temp_c = -4.0
    state.adcs.wheel_momentum_nms = 0.13
    state.adcs.wheel_momentum_limit_nms = 0.12

    breakdown = subsystem.step(state=state, dt_s=5.0, action=None)

    return {
        "active_fault_count": float(breakdown.active_fault_count),
        "highest_fault_level": float(int(breakdown.highest_fault_level)),
        "safe_mode_latched": 1.0 if breakdown.safe_mode_latched else 0.0,
        "last_fault": float(int(breakdown.last_fault)),
        "newly_added_faults": float(breakdown.newly_added_faults),
        "cleared_faults": float(breakdown.cleared_faults),
    }
