from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Set

from models.enums import (
    Action,
    FaultLevel,
    FrameClass,
    GroundPassState,
    LinkQuality,
    PayloadMode,
    PointingQuality,
    SpacecraftMode,
    TargetOpportunity,
    ThermalMode,
)


@dataclass(frozen=True)
class BaselinePolicyConfig:
    min_battery_for_science_pct: float = 40.0
    min_battery_for_capture_pct: float = 50.0
    min_battery_for_downlink_pct: float = 35.0

    low_battery_charge_priority_pct: float = 45.0
    exit_conservative_charge_pct: float = 60.0

    high_temp_c: float = 60.0
    low_temp_c: float = -5.0
    max_memory_fill_ratio_for_capture: float = 0.90

    compress_when_queue_above_mb: float = 64.0
    prefer_high_rate_min_queue_mb: float = 32.0

    discard_cloudy_frames: bool = True
    compress_before_downlink: bool = False

    # anti-stuck recovery
    max_repeat_desat: int = 3
    max_repeat_detumble: int = 3
    max_repeat_fault_recovery: int = 2


@dataclass(frozen=True)
class BaselineDecision:
    action: Action
    reason: str
    priority: str


class BaselinePolicy:
    def __init__(self, config: Optional[BaselinePolicyConfig] = None) -> None:
        self.config = config or BaselinePolicyConfig()
        self._last_action: Optional[Action] = None
        self._repeat_count: int = 0

    # =====================================================
    # Public API
    # =====================================================

    def decide(self, state, valid_actions: Sequence[Action]) -> BaselineDecision:
        valid = self._normalize_valid_actions(valid_actions)
        if not valid:
            raise RuntimeError("BaselinePolicy.decide() received empty valid_actions.")

        battery_pct = self._battery_pct(state)
        payload_temp_c = self._payload_temp_c(state)
        battery_temp_c = self._battery_temp_c(state)
        memory_used_mb = self._memory_used_mb(state)
        memory_capacity_mb = self._memory_capacity_mb(state)
        queue_mb = self._queue_mb(state)

        memory_fill_ratio = memory_used_mb / max(memory_capacity_mb, 1.0)

        spacecraft_mode = self._spacecraft_mode(state)
        payload_mode = self._payload_mode(state)
        frame_class = self._frame_class(state)
        pass_state = self._ground_pass_state(state)
        link_quality = self._link_quality(state)
        target_opportunity = self._target_opportunity(state)
        pointing_quality = self._pointing_quality(state)
        thermal_mode = self._thermal_mode(state)
        fault_level = self._fault_level(state)

        has_frame = frame_class != FrameClass.NONE
        useful_frame = frame_class in {FrameClass.CLEAR, FrameClass.HIGH_VALUE_CLEAR}
        cloudy_frame = frame_class in {FrameClass.CLOUDY, FrameClass.PARTLY_CLOUDY}

        pass_active = pass_state in {
            GroundPassState.ACQUIRE,
            GroundPassState.LOW_ELEVATION,
            GroundPassState.MID_ELEVATION,
            GroundPassState.HIGH_ELEVATION,
            GroundPassState.LOSS,
        }
        strong_pass = pass_state in {
            GroundPassState.MID_ELEVATION,
            GroundPassState.HIGH_ELEVATION,
        }

        link_usable = link_quality in {
            LinkQuality.MARGINAL,
            LinkQuality.GOOD,
            LinkQuality.EXCELLENT,
        }
        link_good = link_quality in {LinkQuality.GOOD, LinkQuality.EXCELLENT}

        target_valid = target_opportunity in {
            TargetOpportunity.VALID,
            TargetOpportunity.HIGH_VALUE,
        }

        payload_ready = payload_mode == PayloadMode.READY
        payload_warming = payload_mode == PayloadMode.WARMUP
        pointing_good_for_imaging = pointing_quality in {
            PointingQuality.USABLE,
            PointingQuality.PRECISE,
        }
        pointing_bad = pointing_quality == PointingQuality.INVALID

        too_hot = payload_temp_c is not None and payload_temp_c >= self.config.high_temp_c
        too_cold = payload_temp_c is not None and payload_temp_c <= self.config.low_temp_c
        heaters_active = self._heaters_active(state, thermal_mode)

        wheel_sat = self._wheel_saturation_high(state)
        needs_detumble = self._needs_detumble(state)
        adcs_pointing_lost = self._adcs_pointing_lost(state)

        a_no_op = self._if_valid(Action.NO_OP, valid)
        a_enter_safe = self._if_valid(Action.ENTER_SAFE_MODE, valid)
        a_detumble = self._if_valid(Action.DETUMBLE, valid)

        a_sun_point_charge = self._if_valid(Action.SUN_POINT_CHARGE, valid)
        a_nadir_point = self._if_valid(Action.NADIR_POINT_STANDBY, valid)
        a_slew_ground = self._if_valid(Action.SLEW_TO_GROUND, valid)
        a_hold_inertial = self._if_valid(Action.HOLD_INERTIAL, valid)
        a_desat = self._if_valid(Action.DESATURATE_WHEELS, valid)

        a_payload_warmup = self._if_valid(Action.PAYLOAD_WARMUP, valid)
        a_capture = self._if_valid(Action.CAPTURE_IMAGE, valid)
        a_classifier = self._if_valid(Action.RUN_CLASSIFIER, valid)
        a_store = self._if_valid(Action.STORE_FRAME, valid)
        a_discard = self._if_valid(Action.DISCARD_FRAME, valid)
        a_compress = self._if_valid(Action.COMPRESS_DATA, valid)

        a_prepare_downlink = self._if_valid(Action.PREPARE_DOWNLINK, valid)
        a_downlink_low = self._if_valid(Action.DOWNLINK_LOW_RATE, valid)
        a_downlink_high = self._if_valid(Action.DOWNLINK_HIGH_RATE, valid)
        a_beacon = self._if_valid(Action.SEND_BEACON, valid)

        a_batt_heater = self._if_valid(Action.ENABLE_BATTERY_HEATER, valid)
        a_payload_heater = self._if_valid(Action.ENABLE_PAYLOAD_HEATER, valid)
        a_disable_heaters = self._if_valid(Action.DISABLE_HEATERS, valid)

        a_fault_recovery = self._if_valid(Action.FAULT_RECOVERY, valid)
        a_reset_payload = self._if_valid(Action.RESET_PAYLOAD, valid)
        a_reset_comms = self._if_valid(Action.RESET_COMMS, valid)
        a_reset_adcs = self._if_valid(Action.RESET_ADCS, valid)
        a_reset_cdh = self._if_valid(Action.RESET_CDH, valid)

        # 1. hard safety
        if too_hot and a_enter_safe is not None:
            return self._finalize(Action.ENTER_SAFE_MODE, "Payload too hot; entering safe mode.", "CRITICAL_SAFETY")

        if battery_pct <= self.config.min_battery_for_downlink_pct and a_enter_safe is not None:
            return self._finalize(Action.ENTER_SAFE_MODE, "Battery critically low; entering safe mode.", "CRITICAL_SAFETY")

        # 2. ADCS recovery ladder
        adcs_unhealthy = wheel_sat or needs_detumble or adcs_pointing_lost or pointing_bad

        if adcs_unhealthy:
            if wheel_sat and a_desat is not None and not self._action_repeated_too_much(Action.DESATURATE_WHEELS, self.config.max_repeat_desat):
                return self._finalize(
                    Action.DESATURATE_WHEELS,
                    "ADCS unhealthy and wheel saturation detected; desaturating wheels.",
                    "ADCS_MAINTENANCE",
                )

            if (needs_detumble or pointing_bad or adcs_pointing_lost) and a_detumble is not None and not self._action_repeated_too_much(Action.DETUMBLE, self.config.max_repeat_detumble):
                return self._finalize(
                    Action.DETUMBLE,
                    "ADCS unhealthy and pointing invalid/lost; attempting detumble.",
                    "ADCS_RECOVERY",
                )

            if a_reset_adcs is not None:
                return self._finalize(
                    Action.RESET_ADCS,
                    "ADCS remains unhealthy after corrective actions; resetting ADCS.",
                    "ADCS_RESET",
                )

            if a_fault_recovery is not None and not self._action_repeated_too_much(Action.FAULT_RECOVERY, self.config.max_repeat_fault_recovery):
                return self._finalize(
                    Action.FAULT_RECOVERY,
                    "ADCS still unhealthy after escalation; attempting generic fault recovery.",
                    "FAULT_RECOVERY",
                )

        # 3. generic critical fault recovery
        if fault_level is not None and fault_level >= FaultLevel.CRITICAL:
            if a_fault_recovery is not None:
                return self._finalize(
                    Action.FAULT_RECOVERY,
                    f"Critical fault level detected ({fault_level.name}); attempting recovery.",
                    "FAULT_RECOVERY",
                )
            if a_enter_safe is not None:
                return self._finalize(
                    Action.ENTER_SAFE_MODE,
                    f"Critical fault level detected ({fault_level.name}); entering safe mode.",
                    "FAULT_RECOVERY",
                )

        # 4. thermal protection
        if battery_temp_c is not None and battery_temp_c <= self.config.low_temp_c and a_batt_heater is not None:
            return self._finalize(Action.ENABLE_BATTERY_HEATER, "Battery too cold; enabling battery heater.", "THERMAL_PROTECT")

        if payload_temp_c is not None and payload_temp_c <= self.config.low_temp_c and a_payload_heater is not None:
            return self._finalize(Action.ENABLE_PAYLOAD_HEATER, "Payload too cold; enabling payload heater.", "THERMAL_PROTECT")

        # 5. frame handling
        if has_frame:
            if frame_class == FrameClass.NONE and a_classifier is not None:
                return self._finalize(Action.RUN_CLASSIFIER, "Frame present but not classified; running classifier.", "CLASSIFY")

            if useful_frame and a_store is not None:
                return self._finalize(Action.STORE_FRAME, f"Useful frame ({frame_class.name}) available; storing.", "STORE_USEFUL")

            if cloudy_frame and self.config.discard_cloudy_frames and a_discard is not None:
                return self._finalize(Action.DISCARD_FRAME, f"Cloudy frame ({frame_class.name}); discarding.", "DISCARD_BAD")

            if a_no_op is not None:
                return self._finalize(Action.NO_OP, "Frame exists but no valid frame-handling action available.", "HOLD")

        # 6. low battery behavior
        if battery_pct < self.config.low_battery_charge_priority_pct:
            if a_sun_point_charge is not None:
                return self._finalize(Action.SUN_POINT_CHARGE, f"Battery low ({battery_pct:.2f}%); charging.", "CHARGE_PRIORITY")
            if a_no_op is not None:
                return self._finalize(Action.NO_OP, f"Battery low ({battery_pct:.2f}%); holding conservative posture.", "CHARGE_PRIORITY")

        # 7. downlink
        if pass_active and queue_mb > 0.0 and battery_pct >= self.config.min_battery_for_downlink_pct:
            if self.config.compress_before_downlink and queue_mb >= self.config.compress_when_queue_above_mb and a_compress is not None:
                return self._finalize(Action.COMPRESS_DATA, f"Queue is large ({queue_mb:.2f} MB); compressing.", "COMPRESS")

            if spacecraft_mode != SpacecraftMode.DOWNLINK and a_prepare_downlink is not None:
                return self._finalize(Action.PREPARE_DOWNLINK, f"Pass active with queued data ({queue_mb:.2f} MB); preparing downlink.", "PREPARE_DOWNLINK")

            if strong_pass and link_good and queue_mb >= self.config.prefer_high_rate_min_queue_mb and a_downlink_high is not None:
                return self._finalize(Action.DOWNLINK_HIGH_RATE, "Strong pass and good link; using high-rate downlink.", "DOWNLINK_HIGH")

            if link_usable and a_downlink_low is not None:
                return self._finalize(Action.DOWNLINK_LOW_RATE, "Pass active with usable link; downlinking low rate.", "DOWNLINK_LOW")

            if a_beacon is not None:
                return self._finalize(Action.SEND_BEACON, "Pass active but no data downlink feasible; sending beacon.", "BEACON")

        # 8. science
        science_allowed = (
            battery_pct >= self.config.min_battery_for_science_pct
            and not too_hot
            and not too_cold
            and memory_fill_ratio <= self.config.max_memory_fill_ratio_for_capture
            and target_valid
            and not adcs_unhealthy
        )

        if science_allowed:
            if not pointing_good_for_imaging:
                if a_nadir_point is not None:
                    return self._finalize(Action.NADIR_POINT_STANDBY, "Target available but pointing insufficient; commanding nadir pointing.", "POINT_FOR_IMAGING")
                if a_hold_inertial is not None:
                    return self._finalize(Action.HOLD_INERTIAL, "Target available but pointing insufficient; holding inertial attitude.", "POINT_FOR_IMAGING")

            if not payload_ready and not payload_warming and a_payload_warmup is not None:
                return self._finalize(Action.PAYLOAD_WARMUP, f"Target opportunity is {target_opportunity.name}; warming payload.", "WARMUP")

            if payload_ready and pointing_good_for_imaging and battery_pct >= self.config.min_battery_for_capture_pct and a_capture is not None:
                return self._finalize(Action.CAPTURE_IMAGE, f"Science opportunity available ({target_opportunity.name}); capturing image.", "CAPTURE")

        # 9. thermal cleanup
        if heaters_active and a_disable_heaters is not None:
            payload_ok = (payload_temp_c is None) or (payload_temp_c > self.config.low_temp_c + 5.0)
            battery_ok = (battery_temp_c is None) or (battery_temp_c > self.config.low_temp_c + 5.0)
            if payload_ok and battery_ok:
                return self._finalize(Action.DISABLE_HEATERS, "Heaters active but temperatures are now fine; disabling heaters.", "THERMAL_CLEANUP")

        # 10. standby
        if battery_pct < self.config.exit_conservative_charge_pct and a_sun_point_charge is not None:
            return self._finalize(Action.SUN_POINT_CHARGE, f"Battery below preferred reserve ({battery_pct:.2f}%); topping up charge.", "TOP_UP_CHARGE")

        if target_valid and a_nadir_point is not None:
            return self._finalize(Action.NADIR_POINT_STANDBY, "Target may soon become exploitable; maintaining nadir standby.", "SCIENCE_STANDBY")

        if pass_active and a_slew_ground is not None:
            return self._finalize(Action.SLEW_TO_GROUND, "Ground pass active; orienting toward ground station.", "PASS_STANDBY")

        if spacecraft_mode in {SpacecraftMode.SAFE, SpacecraftMode.FAULT_RECOVERY, SpacecraftMode.SURVIVAL}:
            for act, reason, prio in [
                (a_reset_payload, "Spacecraft in recovery-like mode; trying payload reset.", "SUBSYSTEM_RECOVERY"),
                (a_reset_comms, "Spacecraft in recovery-like mode; trying comms reset.", "SUBSYSTEM_RECOVERY"),
                (a_reset_cdh, "Spacecraft in recovery-like mode; trying CDH reset.", "SUBSYSTEM_RECOVERY"),
            ]:
                if act is not None:
                    return self._finalize(act, reason, prio)

        if a_no_op is not None:
            return self._finalize(
                Action.NO_OP,
                f"No stronger valid action. battery={battery_pct:.2f}%, target={target_opportunity.name}, pass={pass_state.name}, queue={queue_mb:.2f} MB",
                "NO_OP",
            )

        fallback = next(iter(valid))
        return self._finalize(fallback, "Fallback to first valid action.", "FALLBACK")

    # =====================================================
    # Internal helpers
    # =====================================================

    def _finalize(self, action: Action, reason: str, priority: str) -> BaselineDecision:
        if self._last_action == action:
            self._repeat_count += 1
        else:
            self._last_action = action
            self._repeat_count = 1
        return BaselineDecision(action=action, reason=reason, priority=priority)

    def _action_repeated_too_much(self, action: Action, max_repeat: int) -> bool:
        return self._last_action == action and self._repeat_count >= max_repeat

    def _normalize_valid_actions(self, valid_actions: Sequence[Action]) -> Set[Action]:
        return {a for a in valid_actions if isinstance(a, Action)}

    def _if_valid(self, action: Action, valid: Set[Action]) -> Optional[Action]:
        return action if action in valid else None

    def _get_path(self, obj, *path, default=None):
        cur = obj
        for part in path:
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(part, None)
            elif hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                return default
        return default if cur is None else cur

    def _battery_pct(self, state) -> float:
        for path in [
            ("power", "battery_soc_pct"),
            ("power", "soc_pct"),
            ("eps", "battery_soc_pct"),
            ("battery_soc_pct",),
            ("battery_pct",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None:
                return float(val)
        return 100.0

    def _payload_temp_c(self, state) -> Optional[float]:
        for path in [
            ("thermal", "payload_temp_c"),
            ("thermal", "temperature_c"),
            ("payload_temp_c",),
            ("temperature_c",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None:
                return float(val)
        return None

    def _battery_temp_c(self, state) -> Optional[float]:
        for path in [
            ("thermal", "battery_temp_c"),
            ("battery_temp_c",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None:
                return float(val)
        return None

    def _memory_used_mb(self, state) -> float:
        for path in [
            ("cdh", "memory_used_mb"),
            ("memory_used_mb",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None:
                return float(val)
        return 0.0

    def _memory_capacity_mb(self, state) -> float:
        for path in [
            ("cdh", "memory_capacity_mb"),
            ("memory_capacity_mb",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None:
                return max(1.0, float(val))
        return 1.0

    def _queue_mb(self, state) -> float:
        for path in [
            ("cdh", "downlink_queue_mb"),
            ("comms", "downlink_queue_mb"),
            ("downlink_queue_mb",),
            ("queue_mb",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None:
                return float(val)
        return 0.0

    def _spacecraft_mode(self, state) -> SpacecraftMode:
        for path in [
            ("spacecraft", "mode"),
            ("mode",),
            ("spacecraft_mode",),
        ]:
            val = self._get_path(state, *path, default=None)
            if isinstance(val, SpacecraftMode):
                return val
            if val is not None:
                try:
                    return SpacecraftMode(int(val))
                except Exception:
                    pass
        return SpacecraftMode.NOMINAL

    def _payload_mode(self, state) -> PayloadMode:
        for path in [
            ("payload", "mode"),
            ("payload_mode",),
        ]:
            val = self._get_path(state, *path, default=None)
            if isinstance(val, PayloadMode):
                return val
            if val is not None:
                try:
                    return PayloadMode(int(val))
                except Exception:
                    pass
        return PayloadMode.OFF

    def _frame_class(self, state) -> FrameClass:
        for path in [
            ("payload", "frame_class"),
            ("frame_class",),
        ]:
            val = self._get_path(state, *path, default=None)
            if isinstance(val, FrameClass):
                return val
            if val is not None:
                try:
                    return FrameClass(int(val))
                except Exception:
                    pass
        return FrameClass.NONE

    def _ground_pass_state(self, state) -> GroundPassState:
        for path in [
            ("orbit", "ground_pass_state"),
            ("comms", "ground_pass_state"),
            ("ground_pass_state",),
        ]:
            val = self._get_path(state, *path, default=None)
            if isinstance(val, GroundPassState):
                return val
            if val is not None:
                try:
                    return GroundPassState(int(val))
                except Exception:
                    pass
        return GroundPassState.NONE

    def _link_quality(self, state) -> LinkQuality:
        for path in [
            ("comms", "link_quality"),
            ("link_quality",),
        ]:
            val = self._get_path(state, *path, default=None)
            if isinstance(val, LinkQuality):
                return val
            if val is not None:
                try:
                    return LinkQuality(int(val))
                except Exception:
                    pass
        return LinkQuality.NONE

    def _target_opportunity(self, state) -> TargetOpportunity:
        for path in [
            ("orbit", "target_opportunity"),
            ("target_opportunity",),
        ]:
            val = self._get_path(state, *path, default=None)
            if isinstance(val, TargetOpportunity):
                return val
            if val is not None:
                try:
                    return TargetOpportunity(int(val))
                except Exception:
                    pass
        return TargetOpportunity.NONE

    def _pointing_quality(self, state) -> PointingQuality:
        for path in [
            ("adcs", "pointing_quality"),
            ("pointing_quality",),
        ]:
            val = self._get_path(state, *path, default=None)
            if isinstance(val, PointingQuality):
                return val
            if val is not None:
                try:
                    return PointingQuality(int(val))
                except Exception:
                    pass
        return PointingQuality.INVALID

    def _thermal_mode(self, state) -> Optional[ThermalMode]:
        for path in [
            ("thermal", "mode"),
            ("thermal_mode",),
        ]:
            val = self._get_path(state, *path, default=None)
            if isinstance(val, ThermalMode):
                return val
            if val is not None:
                try:
                    return ThermalMode(int(val))
                except Exception:
                    pass
        return None

    def _fault_level(self, state) -> Optional[FaultLevel]:
        for path in [
            ("faults", "max_fault_level"),
            ("faults", "fault_level"),
            ("max_fault_level",),
            ("fault_level",),
        ]:
            val = self._get_path(state, *path, default=None)
            if isinstance(val, FaultLevel):
                return val
            if val is not None:
                try:
                    return FaultLevel(int(val))
                except Exception:
                    pass
        return None

    def _wheel_saturation_high(self, state) -> bool:
        for path in [
            ("adcs", "wheel_saturation"),
            ("adcs", "wheel_momentum_high"),
            ("wheel_saturation",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None:
                return bool(val)
        return False

    def _needs_detumble(self, state) -> bool:
        for path in [
            ("adcs", "needs_detumble"),
            ("needs_detumble",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None:
                return bool(val)

        for path in [
            ("adcs", "mode"),
            ("attitude_mode",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None and hasattr(val, "name") and val.name == "DETUMBLE":
                return True
        return False

    def _adcs_pointing_lost(self, state) -> bool:
        for path in [
            ("faults", "active_faults"),
            ("faults_active",),
        ]:
            faults = self._get_path(state, *path, default=None)
            if faults:
                try:
                    for f in faults:
                        if isinstance(f, dict):
                            fault_type = f.get("fault_type", None)
                            if hasattr(fault_type, "name") and fault_type.name == "ADCS_POINTING_LOST":
                                return True
                            if "ADCS_POINTING_LOST" in str(fault_type):
                                return True
                        else:
                            if "ADCS_POINTING_LOST" in str(f):
                                return True
                except TypeError:
                    pass
        return False

    def _heaters_active(self, state, thermal_mode: Optional[ThermalMode]) -> bool:
        for path in [
            ("thermal", "battery_heater_on"),
            ("thermal", "payload_heater_on"),
            ("battery_heater_on",),
            ("payload_heater_on",),
        ]:
            val = self._get_path(state, *path, default=None)
            if val is not None and bool(val):
                return True

        if thermal_mode is not None and thermal_mode in {
            ThermalMode.BATTERY_HEATING,
            ThermalMode.PAYLOAD_HEATING,
        }:
            return True
        return False