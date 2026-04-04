from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .enums import (
    Action,
    AttitudeMode,
    CommsMode,
    DataHandlingMode,
    FaultLevel,
    FaultType,
    GroundPassState,
    LinkQuality,
    PayloadMode,
    PointingQuality,
    PowerMode,
    SpacecraftMode,
    SubsystemName,
    SunlightState,
    TargetOpportunity,
    TemperatureBand,
    ThermalMode,
)
from .state_models import SpacecraftState


# =========================================================
# Validation result structures
# =========================================================

@dataclass
class ActionValidationResult:
    """
    Result of checking whether an action can be executed right now.
    """
    valid: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        self.reasons.append(reason)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)


@dataclass
class ActionEffectProfile:
    """
    Nominal operational footprint of an action.

    These values do not directly mutate the state by themselves.
    They provide a standardized description that the environment
    can use while applying dynamics.
    """
    duration_s: float
    extra_load_w: float
    heat_delta_c: float = 0.0
    wheel_momentum_delta_nms: float = 0.0
    memory_delta_mb: float = 0.0
    queue_delta_mb: float = 0.0
    requires_pointing_settle: bool = False
    radio_use: bool = False
    payload_use: bool = False


@dataclass
class ActionSpec:
    """
    Full definition of an action in the simulator.
    """
    action: Action
    display_name: str
    description: str

    allowed_modes: Tuple[SpacecraftMode, ...]
    duration_s: float
    nominal_extra_load_w: float

    validator: Callable[[SpacecraftState], ActionValidationResult]

    heat_delta_c: float = 0.0
    wheel_momentum_delta_nms: float = 0.0
    memory_delta_mb: float = 0.0
    queue_delta_mb: float = 0.0

    requires_pointing_settle: bool = False
    radio_use: bool = False
    payload_use: bool = False
    can_be_interrupted: bool = True
    emergency_action: bool = False

    def profile(self) -> ActionEffectProfile:
        return ActionEffectProfile(
            duration_s=self.duration_s,
            extra_load_w=self.nominal_extra_load_w,
            heat_delta_c=self.heat_delta_c,
            wheel_momentum_delta_nms=self.wheel_momentum_delta_nms,
            memory_delta_mb=self.memory_delta_mb,
            queue_delta_mb=self.queue_delta_mb,
            requires_pointing_settle=self.requires_pointing_settle,
            radio_use=self.radio_use,
            payload_use=self.payload_use,
        )

    def validate(self, state: SpacecraftState) -> ActionValidationResult:
        result = self.validator(state)

        if state.mode not in self.allowed_modes:
            result.add_reason(
                f"Action {self.action.name} not allowed in spacecraft mode {state.mode.name}."
            )

        return result


# =========================================================
# Common helper predicates
# =========================================================

def _blank_valid() -> ActionValidationResult:
    return ActionValidationResult(valid=True)


def _spacecraft_alive_check(state: SpacecraftState, result: ActionValidationResult) -> None:
    if not state.alive or state.done:
        result.add_reason("Spacecraft is no longer operational for action execution.")


def _critical_fault_check(state: SpacecraftState, result: ActionValidationResult) -> None:
    if state.critical_fault:
        result.add_warning("Critical fault is active; only survival/recovery actions are advisable.")


def _thermal_survival_check(state: SpacecraftState, result: ActionValidationResult) -> None:
    if state.thermal.thermal_violation:
        result.add_warning("Thermal violation active; non-survival actions are risky.")


def _require_sunlit(state: SpacecraftState, result: ActionValidationResult) -> None:
    if state.orbit.sunlight_state != SunlightState.SUNLIT:
        result.add_reason("Action requires sunlight, but spacecraft is currently in eclipse.")


def _require_target_available(state: SpacecraftState, result: ActionValidationResult) -> None:
    if not state.orbit.over_target:
        result.add_reason("No imaging target is currently available.")
    if state.orbit.target_opportunity == TargetOpportunity.NONE:
        result.add_reason("Target opportunity is not valid at this step.")


def _require_daylight_imaging(state: SpacecraftState, result: ActionValidationResult) -> None:
    if not state.orbit.day_imaging_valid:
        result.add_reason("Daylight imaging condition is not satisfied.")


def _require_gs_visible(state: SpacecraftState, result: ActionValidationResult) -> None:
    if not state.comms.gs_visible:
        result.add_reason("Ground station is not visible.")
    if state.comms.pass_state == GroundPassState.NONE:
        result.add_reason("No active ground pass is available.")


def _require_link(state: SpacecraftState, result: ActionValidationResult) -> None:
    _require_gs_visible(state, result)
    if not state.comms.link_available:
        result.add_reason("Communications link is not available.")
    elif state.comms.link_quality in (LinkQuality.NONE, LinkQuality.POOR):
        result.add_warning("Link is weak; downlink may be inefficient or fail.")


def _require_payload_ready(state: SpacecraftState, result: ActionValidationResult) -> None:
    if not state.payload.payload_enabled:
        result.add_reason("Payload is not enabled.")
    if state.payload.mode not in (PayloadMode.READY, PayloadMode.IMAGING, PayloadMode.PROCESSING):
        result.add_reason(f"Payload mode {state.payload.mode.name} is not ready for this action.")


def _require_payload_warmable(state: SpacecraftState, result: ActionValidationResult) -> None:
    if state.thermal.payload_temp_band == TemperatureBand.TOO_HOT:
        result.add_reason("Payload is too hot for warmup/activation.")
    if state.faults.has_fault(FaultType.PAYLOAD_OVERTEMP):
        result.add_reason("Payload overtemperature fault is active.")


def _require_frame_present(state: SpacecraftState, result: ActionValidationResult) -> None:
    if not state.payload.has_frame:
        result.add_reason("No current frame is available.")


def _require_frame_not_present(state: SpacecraftState, result: ActionValidationResult) -> None:
    if state.payload.has_frame:
        result.add_reason("A frame is already held by the payload; clear or store it first.")


def _require_memory_margin(
    state: SpacecraftState,
    result: ActionValidationResult,
    required_mb: float,
) -> None:
    if state.cdh.memory_free_mb < required_mb:
        result.add_reason(
            f"Insufficient onboard memory. Required {required_mb:.1f} MB, "
            f"available {state.cdh.memory_free_mb:.1f} MB."
        )


def _require_queue_present(state: SpacecraftState, result: ActionValidationResult) -> None:
    if state.cdh.downlink_queue_mb <= 0.0:
        result.add_reason("Downlink queue is empty.")


def _require_battery_soc(state: SpacecraftState, result: ActionValidationResult, min_soc_pct: float) -> None:
    if state.eps.battery_soc_pct < min_soc_pct:
        result.add_reason(
            f"Battery state of charge too low. "
            f"Required >= {min_soc_pct:.1f}%, current {state.eps.battery_soc_pct:.1f}%."
        )


def _require_battery_temp_nominalish(state: SpacecraftState, result: ActionValidationResult) -> None:
    band = state.thermal.battery_temp_band
    if band in (TemperatureBand.TOO_COLD, TemperatureBand.TOO_HOT):
        result.add_reason(f"Battery thermal band {band.name} prevents this action.")
    elif band in (TemperatureBand.COLD, TemperatureBand.WARM):
        result.add_warning(f"Battery thermal band {band.name} is non-nominal.")


def _require_pointing_quality(
    state: SpacecraftState,
    result: ActionValidationResult,
    minimum: PointingQuality,
) -> None:
    if state.adcs.pointing_quality < minimum:
        result.add_reason(
            f"Pointing quality {state.adcs.pointing_quality.name} is below required "
            f"{minimum.name}."
        )


def _require_attitude_mode(
    state: SpacecraftState,
    result: ActionValidationResult,
    allowed: Tuple[AttitudeMode, ...],
) -> None:
    if state.adcs.mode not in allowed:
        allowed_str = ", ".join(a.name for a in allowed)
        result.add_reason(
            f"Attitude mode {state.adcs.mode.name} invalid; required one of [{allowed_str}]."
        )


def _warn_if_wheels_high(state: SpacecraftState, result: ActionValidationResult) -> None:
    if state.adcs.wheels_saturated:
        result.add_warning("Reaction wheels are saturated; precision actions may degrade.")


def _require_comms_not_faulted(state: SpacecraftState, result: ActionValidationResult) -> None:
    if state.faults.has_fault(FaultType.COMMS_LINK_LOSS):
        result.add_warning("Active communications link-loss fault present.")
    if state.faults.has_fault(FaultType.COMMS_TIMEOUT):
        result.add_warning("Active communications timeout fault present.")


def _require_storage_healthy(state: SpacecraftState, result: ActionValidationResult) -> None:
    if not state.cdh.filesystem_healthy or state.cdh.storage_corrupted:
        result.add_reason("Storage system is not healthy.")


def _warn_if_power_negative(state: SpacecraftState, result: ActionValidationResult) -> None:
    if not state.power_positive:
        result.add_warning("Current power balance is negative.")


def _warn_if_memory_high(state: SpacecraftState, result: ActionValidationResult) -> None:
    if state.memory_pressure.value >= 2:
        result.add_warning(f"Memory pressure is {state.memory_pressure.name}.")


# =========================================================
# Validators for each action
# =========================================================

def validate_no_op(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    return result


def validate_enter_safe_mode(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    return result


def validate_detumble(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)

    if state.adcs.body_rate_deg_s < 0.1:
        result.add_warning("Body rates already low; detumble may have limited benefit.")

    return result


def validate_sun_point_charge(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_sunlit(state, result)
    _warn_if_memory_high(state, result)

    if state.adcs.mode == AttitudeMode.SUN_POINTING and state.adcs.settled:
        result.add_warning("Already in settled sun-pointing posture.")

    return result


def validate_nadir_point_standby(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)

    if state.orbit.target_opportunity == TargetOpportunity.NONE:
        result.add_warning("No current target; nadir standby may consume power without immediate science gain.")

    _require_battery_soc(state, result, min_soc_pct=15.0)
    _require_battery_temp_nominalish(state, result)
    _warn_if_wheels_high(state, result)

    return result


def validate_slew_to_ground(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_gs_visible(state, result)
    _require_battery_soc(state, result, min_soc_pct=15.0)
    _require_battery_temp_nominalish(state, result)
    _warn_if_wheels_high(state, result)
    _require_comms_not_faulted(state, result)
    return result


def validate_hold_inertial(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_battery_soc(state, result, min_soc_pct=12.0)
    _warn_if_wheels_high(state, result)
    return result


def validate_desaturate_wheels(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)

    if not state.adcs.wheels_saturated and state.adcs.wheel_momentum_nms < 0.7 * state.adcs.wheel_momentum_limit_nms:
        result.add_warning("Wheel momentum is not especially high; desaturation may be unnecessary.")

    _require_battery_soc(state, result, min_soc_pct=12.0)
    return result


def validate_payload_warmup(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_payload_warmable(state, result)
    _require_battery_soc(state, result, min_soc_pct=18.0)
    _require_battery_temp_nominalish(state, result)

    if state.payload.mode in (PayloadMode.READY, PayloadMode.IMAGING):
        result.add_warning("Payload is already warm or active.")

    return result


def validate_capture_image(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)

    _require_target_available(state, result)
    _require_daylight_imaging(state, result)
    _require_payload_ready(state, result)
    _require_frame_not_present(state, result)
    _require_memory_margin(state, result, required_mb=32.0)
    _require_battery_soc(state, result, min_soc_pct=22.0)
    _require_battery_temp_nominalish(state, result)
    _require_attitude_mode(
        state,
        result,
        allowed=(AttitudeMode.NADIR_POINTING, AttitudeMode.GROUND_TRACKING, AttitudeMode.INERTIAL_HOLD),
    )
    _require_pointing_quality(state, result, minimum=PointingQuality.USABLE)
    _warn_if_wheels_high(state, result)
    _warn_if_power_negative(state, result)

    if state.thermal.payload_temp_band in (TemperatureBand.COLD, TemperatureBand.WARM):
        result.add_warning("Payload thermal state is non-nominal for best image quality.")

    return result


def validate_run_classifier(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)

    _require_frame_present(state, result)
    _require_payload_ready(state, result)
    _require_battery_soc(state, result, min_soc_pct=15.0)
    _require_storage_healthy(state, result)

    if state.payload.current_frame_class != state.payload.current_frame_class.NONE:
        result.add_warning("Frame already has a class label; reclassification may be redundant.")

    return result


def validate_store_frame(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)

    _require_frame_present(state, result)
    _require_storage_healthy(state, result)
    _require_battery_soc(state, result, min_soc_pct=10.0)

    frame_mb = max(1.0, state.payload.current_frame_size_mb)
    _require_memory_margin(state, result, required_mb=frame_mb * state.cdh.compression_ratio)

    if state.payload.current_frame_class == state.payload.current_frame_class.CLOUDY:
        result.add_warning("Current frame is classified as CLOUDY; storing it may waste memory.")
    if state.memory_pressure.value >= 2:
        result.add_warning("Memory pressure is high; storing may increase downlink urgency.")

    return result


def validate_discard_frame(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_frame_present(state, result)

    if state.payload.current_frame_class in (
        state.payload.current_frame_class.CLEAR,
        state.payload.current_frame_class.HIGH_VALUE_CLEAR,
    ):
        result.add_warning("Discarding a potentially useful frame.")

    return result


def validate_compress_data(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_frame_present(state, result)
    _require_storage_healthy(state, result)
    _require_battery_soc(state, result, min_soc_pct=12.0)
    return result


def validate_prepare_downlink(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_gs_visible(state, result)
    _require_queue_present(state, result)
    _require_battery_soc(state, result, min_soc_pct=18.0)
    _require_battery_temp_nominalish(state, result)
    _require_comms_not_faulted(state, result)
    return result


def validate_downlink_low_rate(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_link(state, result)
    _require_queue_present(state, result)
    _require_battery_soc(state, result, min_soc_pct=18.0)
    _require_battery_temp_nominalish(state, result)
    _require_attitude_mode(
        state,
        result,
        allowed=(AttitudeMode.GROUND_TRACKING, AttitudeMode.SLEW_MANEUVER, AttitudeMode.INERTIAL_HOLD),
    )
    _require_comms_not_faulted(state, result)

    if state.comms.link_quality == LinkQuality.EXCELLENT:
        result.add_warning("Link quality is excellent; high-rate downlink may be preferable.")

    return result


def validate_downlink_high_rate(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_link(state, result)
    _require_queue_present(state, result)
    _require_battery_soc(state, result, min_soc_pct=25.0)
    _require_battery_temp_nominalish(state, result)
    _require_attitude_mode(
        state,
        result,
        allowed=(AttitudeMode.GROUND_TRACKING, AttitudeMode.SLEW_MANEUVER),
    )
    _require_comms_not_faulted(state, result)

    if state.comms.link_quality not in (LinkQuality.GOOD, LinkQuality.EXCELLENT):
        result.add_reason("High-rate downlink requires GOOD or EXCELLENT link quality.")

    return result


def validate_send_beacon(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_battery_soc(state, result, min_soc_pct=8.0)

    if not state.comms.gs_visible:
        result.add_warning("Beacon can still be broadcast, but direct pass visibility is absent.")

    return result


def validate_enable_battery_heater(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_battery_soc(state, result, min_soc_pct=10.0)

    if state.thermal.battery_heater_on:
        result.add_warning("Battery heater is already enabled.")
    if state.thermal.battery_temp_band in (TemperatureBand.NOMINAL, TemperatureBand.WARM, TemperatureBand.TOO_HOT):
        result.add_warning("Battery heater may be unnecessary at the current battery temperature.")

    return result


def validate_enable_payload_heater(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_battery_soc(state, result, min_soc_pct=12.0)

    if state.thermal.payload_heater_on:
        result.add_warning("Payload heater is already enabled.")
    if state.thermal.payload_temp_band in (TemperatureBand.NOMINAL, TemperatureBand.WARM, TemperatureBand.TOO_HOT):
        result.add_warning("Payload heater may be unnecessary at the current payload temperature.")

    return result


def validate_disable_heaters(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)

    if not state.thermal.battery_heater_on and not state.thermal.payload_heater_on:
        result.add_warning("No heaters are currently enabled.")

    return result


def validate_fault_recovery(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)

    if state.faults.highest_fault_level == FaultLevel.NONE:
        result.add_warning("No active fault exists; recovery action may be unnecessary.")

    return result


def validate_reset_payload(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_battery_soc(state, result, min_soc_pct=10.0)

    if not (
        state.faults.has_fault(FaultType.PAYLOAD_STUCK)
        or state.faults.has_fault(FaultType.PAYLOAD_OVERTEMP)
        or state.faults.has_fault(FaultType.CLASSIFIER_FAILURE)
    ):
        result.add_warning("No explicit payload-related fault is active.")

    return result


def validate_reset_comms(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_battery_soc(state, result, min_soc_pct=10.0)

    if not (
        state.faults.has_fault(FaultType.COMMS_LINK_LOSS)
        or state.faults.has_fault(FaultType.COMMS_TIMEOUT)
        or state.faults.has_fault(FaultType.DOWNLINK_ABORTED)
    ):
        result.add_warning("No explicit communications fault is active.")

    return result


def validate_reset_adcs(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_battery_soc(state, result, min_soc_pct=12.0)

    if not (
        state.faults.has_fault(FaultType.ADCS_POINTING_LOST)
        or state.faults.has_fault(FaultType.WHEEL_SATURATION)
        or state.faults.has_fault(FaultType.SENSOR_DEGRADED)
    ):
        result.add_warning("No explicit ADCS-related fault is active.")

    return result


def validate_reset_cdh(state: SpacecraftState) -> ActionValidationResult:
    result = _blank_valid()
    _spacecraft_alive_check(state, result)
    _require_battery_soc(state, result, min_soc_pct=10.0)

    if not (
        state.faults.has_fault(FaultType.CDH_RESET_LOOP)
        or state.faults.has_fault(FaultType.STORAGE_CORRUPTION)
        or state.faults.has_fault(FaultType.MEMORY_FULL)
    ):
        result.add_warning("No explicit CDH/storage-related fault is active.")

    return result


# =========================================================
# Action catalog
# =========================================================

ALLOWED_MODES_COMMON_NOMINAL = (
    SpacecraftMode.NOMINAL,
    SpacecraftMode.SCIENCE,
    SpacecraftMode.DOWNLINK,
    SpacecraftMode.ECLIPSE_POWER_SAVE,
    SpacecraftMode.FAULT_RECOVERY,
)

ACTION_SPECS: Dict[Action, ActionSpec] = {
    Action.NO_OP: ActionSpec(
        action=Action.NO_OP,
        display_name="No Operation",
        description="Hold the current posture with no explicit subsystem command beyond normal background control.",
        allowed_modes=(
            SpacecraftMode.BOOT,
            SpacecraftMode.SAFE,
            SpacecraftMode.DETUMBLE,
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.ECLIPSE_POWER_SAVE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=5.0,
        nominal_extra_load_w=0.0,
        validator=validate_no_op,
    ),

    Action.ENTER_SAFE_MODE: ActionSpec(
        action=Action.ENTER_SAFE_MODE,
        display_name="Enter Safe Mode",
        description="Transition spacecraft into a survival-oriented safe configuration.",
        allowed_modes=(
            SpacecraftMode.BOOT,
            SpacecraftMode.SAFE,
            SpacecraftMode.DETUMBLE,
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.ECLIPSE_POWER_SAVE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=5.0,
        nominal_extra_load_w=0.5,
        validator=validate_enter_safe_mode,
        emergency_action=True,
    ),

    Action.DETUMBLE: ActionSpec(
        action=Action.DETUMBLE,
        display_name="Detumble",
        description="Use magnetic control / attitude logic to reduce body rates after disturbance or fault.",
        allowed_modes=(
            SpacecraftMode.BOOT,
            SpacecraftMode.SAFE,
            SpacecraftMode.DETUMBLE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=20.0,
        nominal_extra_load_w=1.2,
        validator=validate_detumble,
        wheel_momentum_delta_nms=-0.005,
    ),

    Action.SUN_POINT_CHARGE: ActionSpec(
        action=Action.SUN_POINT_CHARGE,
        display_name="Sun-Point and Charge",
        description="Point solar arrays for favorable power generation and remain power-positive.",
        allowed_modes=(
            SpacecraftMode.SAFE,
            SpacecraftMode.NOMINAL,
            SpacecraftMode.ECLIPSE_POWER_SAVE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=15.0,
        nominal_extra_load_w=0.8,
        validator=validate_sun_point_charge,
        requires_pointing_settle=True,
    ),

    Action.NADIR_POINT_STANDBY: ActionSpec(
        action=Action.NADIR_POINT_STANDBY,
        display_name="Nadir Point Standby",
        description="Acquire nadir-pointing posture to prepare for Earth observation imaging.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=15.0,
        nominal_extra_load_w=1.5,
        validator=validate_nadir_point_standby,
        heat_delta_c=0.1,
        wheel_momentum_delta_nms=0.006,
        requires_pointing_settle=True,
    ),

    Action.SLEW_TO_GROUND: ActionSpec(
        action=Action.SLEW_TO_GROUND,
        display_name="Slew to Ground Station",
        description="Rotate spacecraft into ground-tracking geometry for an upcoming or active pass.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=20.0,
        nominal_extra_load_w=1.7,
        validator=validate_slew_to_ground,
        heat_delta_c=0.15,
        wheel_momentum_delta_nms=0.008,
        requires_pointing_settle=True,
    ),

    Action.HOLD_INERTIAL: ActionSpec(
        action=Action.HOLD_INERTIAL,
        display_name="Hold Inertial Attitude",
        description="Hold a fixed inertial attitude for calibration, coast, or operational stability.",
        allowed_modes=ALLOWED_MODES_COMMON_NOMINAL,
        duration_s=10.0,
        nominal_extra_load_w=1.0,
        validator=validate_hold_inertial,
        heat_delta_c=0.05,
        wheel_momentum_delta_nms=0.003,
    ),

    Action.DESATURATE_WHEELS: ActionSpec(
        action=Action.DESATURATE_WHEELS,
        display_name="Desaturate Reaction Wheels",
        description="Unload accumulated wheel momentum using magnetic torquers / control logic.",
        allowed_modes=ALLOWED_MODES_COMMON_NOMINAL,
        duration_s=25.0,
        nominal_extra_load_w=1.3,
        validator=validate_desaturate_wheels,
        wheel_momentum_delta_nms=-0.03,
    ),

    Action.PAYLOAD_WARMUP: ActionSpec(
        action=Action.PAYLOAD_WARMUP,
        display_name="Warm Payload",
        description="Bring imaging payload to operational thermal and electrical readiness.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=30.0,
        nominal_extra_load_w=2.0,
        validator=validate_payload_warmup,
        heat_delta_c=0.3,
        payload_use=True,
    ),

    Action.CAPTURE_IMAGE: ActionSpec(
        action=Action.CAPTURE_IMAGE,
        display_name="Capture Image",
        description="Acquire a new Earth observation frame under valid illumination, target, and pointing conditions.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
        ),
        duration_s=8.0,
        nominal_extra_load_w=2.8,
        validator=validate_capture_image,
        heat_delta_c=0.2,
        wheel_momentum_delta_nms=0.005,
        memory_delta_mb=32.0,
        payload_use=True,
    ),

    Action.RUN_CLASSIFIER: ActionSpec(
        action=Action.RUN_CLASSIFIER,
        display_name="Run Onboard Classifier",
        description="Process current frame using onboard inference to estimate cloudiness and usefulness.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=6.0,
        nominal_extra_load_w=2.2,
        validator=validate_run_classifier,
        heat_delta_c=0.12,
        payload_use=True,
    ),

    Action.STORE_FRAME: ActionSpec(
        action=Action.STORE_FRAME,
        display_name="Store Frame",
        description="Persist the current frame into managed onboard storage and enqueue it if useful.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=4.0,
        nominal_extra_load_w=1.0,
        validator=validate_store_frame,
        memory_delta_mb=16.0,
        queue_delta_mb=16.0,
    ),

    Action.DISCARD_FRAME: ActionSpec(
        action=Action.DISCARD_FRAME,
        display_name="Discard Frame",
        description="Delete the current payload frame instead of committing it to storage.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=2.0,
        nominal_extra_load_w=0.2,
        validator=validate_discard_frame,
    ),

    Action.COMPRESS_DATA: ActionSpec(
        action=Action.COMPRESS_DATA,
        display_name="Compress Data",
        description="Apply onboard compression to frame data before storage or transmission.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=5.0,
        nominal_extra_load_w=1.5,
        validator=validate_compress_data,
        heat_delta_c=0.08,
    ),

    Action.PREPARE_DOWNLINK: ActionSpec(
        action=Action.PREPARE_DOWNLINK,
        display_name="Prepare Downlink",
        description="Enable and configure communications stack in anticipation of a data pass.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=8.0,
        nominal_extra_load_w=1.8,
        validator=validate_prepare_downlink,
        radio_use=True,
    ),

    Action.DOWNLINK_LOW_RATE: ActionSpec(
        action=Action.DOWNLINK_LOW_RATE,
        display_name="Low-Rate Downlink",
        description="Transmit queued mission data at a conservative rate with higher tolerance to weak links.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=10.0,
        nominal_extra_load_w=3.2,
        validator=validate_downlink_low_rate,
        heat_delta_c=0.15,
        queue_delta_mb=-0.3,
        radio_use=True,
    ),

    Action.DOWNLINK_HIGH_RATE: ActionSpec(
        action=Action.DOWNLINK_HIGH_RATE,
        display_name="High-Rate Downlink",
        description="Transmit queued mission data at a high rate during strong ground contacts.",
        allowed_modes=(
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=10.0,
        nominal_extra_load_w=4.6,
        validator=validate_downlink_high_rate,
        heat_delta_c=0.25,
        queue_delta_mb=-2.0,
        radio_use=True,
    ),

    Action.SEND_BEACON: ActionSpec(
        action=Action.SEND_BEACON,
        display_name="Send Beacon",
        description="Transmit a lightweight status beacon for health and housekeeping telemetry.",
        allowed_modes=(
            SpacecraftMode.SAFE,
            SpacecraftMode.NOMINAL,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.ECLIPSE_POWER_SAVE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=3.0,
        nominal_extra_load_w=1.4,
        validator=validate_send_beacon,
        radio_use=True,
    ),

    Action.ENABLE_BATTERY_HEATER: ActionSpec(
        action=Action.ENABLE_BATTERY_HEATER,
        display_name="Enable Battery Heater",
        description="Turn on battery heating for eclipse or cold-soak survival management.",
        allowed_modes=(
            SpacecraftMode.SAFE,
            SpacecraftMode.NOMINAL,
            SpacecraftMode.ECLIPSE_POWER_SAVE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=2.0,
        nominal_extra_load_w=1.2,
        validator=validate_enable_battery_heater,
        heat_delta_c=0.3,
    ),

    Action.ENABLE_PAYLOAD_HEATER: ActionSpec(
        action=Action.ENABLE_PAYLOAD_HEATER,
        display_name="Enable Payload Heater",
        description="Heat payload hardware toward operational conditions.",
        allowed_modes=(
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.FAULT_RECOVERY,
        ),
        duration_s=2.0,
        nominal_extra_load_w=1.5,
        validator=validate_enable_payload_heater,
        heat_delta_c=0.35,
    ),

    Action.DISABLE_HEATERS: ActionSpec(
        action=Action.DISABLE_HEATERS,
        display_name="Disable Heaters",
        description="Turn off spacecraft heaters to reduce load or prevent overheating.",
        allowed_modes=(
            SpacecraftMode.SAFE,
            SpacecraftMode.NOMINAL,
            SpacecraftMode.SCIENCE,
            SpacecraftMode.DOWNLINK,
            SpacecraftMode.ECLIPSE_POWER_SAVE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=1.0,
        nominal_extra_load_w=0.1,
        validator=validate_disable_heaters,
    ),

    Action.FAULT_RECOVERY: ActionSpec(
        action=Action.FAULT_RECOVERY,
        display_name="Fault Recovery",
        description="Run generic spacecraft fault recovery workflow.",
        allowed_modes=(
            SpacecraftMode.SAFE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=10.0,
        nominal_extra_load_w=0.8,
        validator=validate_fault_recovery,
        emergency_action=True,
    ),

    Action.RESET_PAYLOAD: ActionSpec(
        action=Action.RESET_PAYLOAD,
        display_name="Reset Payload",
        description="Perform a controlled reset of the payload and associated processing path.",
        allowed_modes=(
            SpacecraftMode.SAFE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=6.0,
        nominal_extra_load_w=0.6,
        validator=validate_reset_payload,
        emergency_action=True,
    ),

    Action.RESET_COMMS: ActionSpec(
        action=Action.RESET_COMMS,
        display_name="Reset Communications",
        description="Perform a controlled communications subsystem reset.",
        allowed_modes=(
            SpacecraftMode.SAFE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=6.0,
        nominal_extra_load_w=0.5,
        validator=validate_reset_comms,
        emergency_action=True,
    ),

    Action.RESET_ADCS: ActionSpec(
        action=Action.RESET_ADCS,
        display_name="Reset ADCS",
        description="Perform a controlled reset of attitude sensors/control chain.",
        allowed_modes=(
            SpacecraftMode.SAFE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=8.0,
        nominal_extra_load_w=0.8,
        validator=validate_reset_adcs,
        emergency_action=True,
    ),

    Action.RESET_CDH: ActionSpec(
        action=Action.RESET_CDH,
        display_name="Reset CDH",
        description="Perform a controlled reset of avionics / command and data handling stack.",
        allowed_modes=(
            SpacecraftMode.SAFE,
            SpacecraftMode.FAULT_RECOVERY,
            SpacecraftMode.SURVIVAL,
        ),
        duration_s=8.0,
        nominal_extra_load_w=0.7,
        validator=validate_reset_cdh,
        emergency_action=True,
    ),
}


# =========================================================
# Public API helpers
# =========================================================

def get_action_spec(action: Action) -> ActionSpec:
    return ACTION_SPECS[action]


def list_all_actions() -> List[Action]:
    return list(ACTION_SPECS.keys())


def validate_action(state: SpacecraftState, action: Action) -> ActionValidationResult:
    return get_action_spec(action).validate(state)


def is_action_valid(state: SpacecraftState, action: Action) -> bool:
    return validate_action(state, action).valid


def get_valid_actions(state: SpacecraftState) -> List[Action]:
    return [action for action in ACTION_SPECS if is_action_valid(state, action)]


def get_emergency_actions() -> List[Action]:
    return [spec.action for spec in ACTION_SPECS.values() if spec.emergency_action]


def get_actions_for_mode(mode: SpacecraftMode) -> List[Action]:
    return [
        spec.action
        for spec in ACTION_SPECS.values()
        if mode in spec.allowed_modes
    ]


def action_mask(state: SpacecraftState, ordered_actions: Optional[List[Action]] = None) -> List[int]:
    """
    Returns a binary mask aligned with ordered_actions.
    Useful for RL algorithms that support invalid-action masking.
    """
    if ordered_actions is None:
        ordered_actions = list_all_actions()

    return [1 if is_action_valid(state, action) else 0 for action in ordered_actions]


def summarize_action_validation(state: SpacecraftState, action: Action) -> Dict[str, object]:
    """
    Human-readable diagnostic for debugging policies.
    """
    spec = get_action_spec(action)
    result = spec.validate(state)

    return {
        "action": action.name,
        "display_name": spec.display_name,
        "valid": result.valid,
        "reasons": result.reasons,
        "warnings": result.warnings,
        "duration_s": spec.duration_s,
        "nominal_extra_load_w": spec.nominal_extra_load_w,
        "requires_pointing_settle": spec.requires_pointing_settle,
        "radio_use": spec.radio_use,
        "payload_use": spec.payload_use,
        "allowed_modes": [m.name for m in spec.allowed_modes],
    }


# =========================================================
# Action transition intent helpers
# =========================================================

def recommended_mode_after_action(action: Action) -> Optional[SpacecraftMode]:
    """
    Returns a suggested top-level spacecraft mode transition for the environment.
    This is not mandatory, but provides a clean hook for cubesat_env.py.
    """
    mapping = {
        Action.ENTER_SAFE_MODE: SpacecraftMode.SAFE,
        Action.DETUMBLE: SpacecraftMode.DETUMBLE,
        Action.SUN_POINT_CHARGE: SpacecraftMode.ECLIPSE_POWER_SAVE,
        Action.NADIR_POINT_STANDBY: SpacecraftMode.SCIENCE,
        Action.SLEW_TO_GROUND: SpacecraftMode.DOWNLINK,
        Action.CAPTURE_IMAGE: SpacecraftMode.SCIENCE,
        Action.RUN_CLASSIFIER: SpacecraftMode.SCIENCE,
        Action.STORE_FRAME: SpacecraftMode.SCIENCE,
        Action.DISCARD_FRAME: SpacecraftMode.SCIENCE,
        Action.COMPRESS_DATA: SpacecraftMode.SCIENCE,
        Action.PREPARE_DOWNLINK: SpacecraftMode.DOWNLINK,
        Action.DOWNLINK_LOW_RATE: SpacecraftMode.DOWNLINK,
        Action.DOWNLINK_HIGH_RATE: SpacecraftMode.DOWNLINK,
        Action.FAULT_RECOVERY: SpacecraftMode.FAULT_RECOVERY,
        Action.RESET_PAYLOAD: SpacecraftMode.FAULT_RECOVERY,
        Action.RESET_COMMS: SpacecraftMode.FAULT_RECOVERY,
        Action.RESET_ADCS: SpacecraftMode.FAULT_RECOVERY,
        Action.RESET_CDH: SpacecraftMode.FAULT_RECOVERY,
    }
    return mapping.get(action, None)


def recommended_attitude_after_action(action: Action) -> Optional[AttitudeMode]:
    mapping = {
        Action.DETUMBLE: AttitudeMode.DETUMBLE,
        Action.SUN_POINT_CHARGE: AttitudeMode.SUN_POINTING,
        Action.NADIR_POINT_STANDBY: AttitudeMode.NADIR_POINTING,
        Action.SLEW_TO_GROUND: AttitudeMode.GROUND_TRACKING,
        Action.HOLD_INERTIAL: AttitudeMode.INERTIAL_HOLD,
        Action.DESATURATE_WHEELS: AttitudeMode.MOMENTUM_DUMP,
        Action.ENTER_SAFE_MODE: AttitudeMode.SAFE_SUN_ACQUIRE,
    }
    return mapping.get(action, None)


def recommended_payload_mode_after_action(action: Action) -> Optional[PayloadMode]:
    mapping = {
        Action.PAYLOAD_WARMUP: PayloadMode.WARMUP,
        Action.CAPTURE_IMAGE: PayloadMode.IMAGING,
        Action.RUN_CLASSIFIER: PayloadMode.PROCESSING,
        Action.STORE_FRAME: PayloadMode.STANDBY,
        Action.DISCARD_FRAME: PayloadMode.STANDBY,
        Action.RESET_PAYLOAD: PayloadMode.OFF,
        Action.ENTER_SAFE_MODE: PayloadMode.OFF,
    }
    return mapping.get(action, None)


def recommended_comms_mode_after_action(action: Action) -> Optional[CommsMode]:
    mapping = {
        Action.PREPARE_DOWNLINK: CommsMode.PASS_PREP,
        Action.DOWNLINK_LOW_RATE: CommsMode.LOW_RATE_TX,
        Action.DOWNLINK_HIGH_RATE: CommsMode.HIGH_RATE_TX,
        Action.SEND_BEACON: CommsMode.BEACON,
        Action.RESET_COMMS: CommsMode.OFF,
        Action.ENTER_SAFE_MODE: CommsMode.LISTEN,
    }
    return mapping.get(action, None)


def recommended_cdh_mode_after_action(action: Action) -> Optional[DataHandlingMode]:
    mapping = {
        Action.RUN_CLASSIFIER: DataHandlingMode.CLASSIFYING,
        Action.STORE_FRAME: DataHandlingMode.BUFFERING,
        Action.DISCARD_FRAME: DataHandlingMode.PURGING,
        Action.COMPRESS_DATA: DataHandlingMode.COMPRESSING,
        Action.PREPARE_DOWNLINK: DataHandlingMode.QUEUEING,
        Action.DOWNLINK_LOW_RATE: DataHandlingMode.DOWNLINKING,
        Action.DOWNLINK_HIGH_RATE: DataHandlingMode.DOWNLINKING,
        Action.RESET_CDH: DataHandlingMode.RECOVERY,
    }
    return mapping.get(action, None)


def recommended_power_mode_after_action(action: Action) -> Optional[PowerMode]:
    mapping = {
        Action.SUN_POINT_CHARGE: PowerMode.CHARGE_PRIORITY,
        Action.ENTER_SAFE_MODE: PowerMode.POWER_SAVE,
        Action.ENABLE_BATTERY_HEATER: PowerMode.ECLIPSE_CONSERVE,
        Action.ENABLE_PAYLOAD_HEATER: PowerMode.NOMINAL,
    }
    return mapping.get(action, None)


def recommended_thermal_mode_after_action(action: Action) -> Optional[ThermalMode]:
    mapping = {
        Action.ENABLE_BATTERY_HEATER: ThermalMode.BATTERY_HEATING,
        Action.ENABLE_PAYLOAD_HEATER: ThermalMode.PAYLOAD_HEATING,
        Action.DISABLE_HEATERS: ThermalMode.NOMINAL,
        Action.ENTER_SAFE_MODE: ThermalMode.SURVIVAL,
    }
    return mapping.get(action, None)


# =========================================================
# Environment-side convenience hooks
# =========================================================

def apply_recommended_mode_transitions(state: SpacecraftState, action: Action) -> None:
    """
    Lightweight helper for cubesat_env.py.
    This updates top-level subsystem modes after an action is accepted.
    """
    new_sc_mode = recommended_mode_after_action(action)
    if new_sc_mode is not None:
        state.mode = new_sc_mode

    new_att_mode = recommended_attitude_after_action(action)
    if new_att_mode is not None:
        state.adcs.mode = new_att_mode

    new_payload_mode = recommended_payload_mode_after_action(action)
    if new_payload_mode is not None:
        state.payload.mode = new_payload_mode

    new_comms_mode = recommended_comms_mode_after_action(action)
    if new_comms_mode is not None:
        state.comms.mode = new_comms_mode

    new_cdh_mode = recommended_cdh_mode_after_action(action)
    if new_cdh_mode is not None:
        state.cdh.mode = new_cdh_mode

    new_power_mode = recommended_power_mode_after_action(action)
    if new_power_mode is not None:
        state.eps.mode = new_power_mode

    new_thermal_mode = recommended_thermal_mode_after_action(action)
    if new_thermal_mode is not None:
        state.thermal.mode = new_thermal_mode


# =========================================================
# Ordered action list for RL consistency
# =========================================================

DEFAULT_ACTION_ORDER: List[Action] = [
    Action.NO_OP,
    Action.ENTER_SAFE_MODE,
    Action.DETUMBLE,
    Action.SUN_POINT_CHARGE,
    Action.NADIR_POINT_STANDBY,
    Action.SLEW_TO_GROUND,
    Action.HOLD_INERTIAL,
    Action.DESATURATE_WHEELS,
    Action.PAYLOAD_WARMUP,
    Action.CAPTURE_IMAGE,
    Action.RUN_CLASSIFIER,
    Action.STORE_FRAME,
    Action.DISCARD_FRAME,
    Action.COMPRESS_DATA,
    Action.PREPARE_DOWNLINK,
    Action.DOWNLINK_LOW_RATE,
    Action.DOWNLINK_HIGH_RATE,
    Action.SEND_BEACON,
    Action.ENABLE_BATTERY_HEATER,
    Action.ENABLE_PAYLOAD_HEATER,
    Action.DISABLE_HEATERS,
    Action.FAULT_RECOVERY,
    Action.RESET_PAYLOAD,
    Action.RESET_COMMS,
    Action.RESET_ADCS,
    Action.RESET_CDH,
]


ACTION_INDEX_MAP: Dict[Action, int] = {action: idx for idx, action in enumerate(DEFAULT_ACTION_ORDER)}
INDEX_ACTION_MAP: Dict[int, Action] = {idx: action for action, idx in ACTION_INDEX_MAP.items()}


def action_to_index(action: Action) -> int:
    return ACTION_INDEX_MAP[action]


def index_to_action(index: int) -> Action:
    return INDEX_ACTION_MAP[index]