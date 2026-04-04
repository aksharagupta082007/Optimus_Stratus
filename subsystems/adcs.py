from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from models.action_space import ActionEffectProfile
from models.enums import Action, AttitudeMode
from models.state_models import SpacecraftState


# =========================================================
# Configuration
# =========================================================

@dataclass(frozen=True)
class ADCSConfig:
    """
    Operational ADCS model for a CubeSat mission simulator.

    This is not a full rigid-body dynamics or Kalman-filter estimator model.
    It is a high-quality operational simulator for mission planning / RL that models:

    - attitude mode transitions
    - slew settling
    - pointing error convergence
    - body-rate reduction / increase
    - reaction-wheel momentum buildup
    - momentum dumping / desaturation
    - coarse lock flags (sun, nadir, ground)
    - sensor availability effects

    Units:
    - pointing error: deg
    - body rate: deg/s
    - wheel momentum: Nms
    - time: s
    """

    # -----------------------------------------------------
    # Initialization defaults
    # -----------------------------------------------------
    initial_pointing_error_deg: float = 8.0
    initial_body_rate_deg_s: float = 1.5
    initial_wheel_momentum_nms: float = 0.03
    wheel_momentum_limit_nms: float = 0.12

    # -----------------------------------------------------
    # Pointing quality thresholds
    # These should match / remain consistent with state_models.py
    # -----------------------------------------------------
    precise_pointing_threshold_deg: float = 0.30
    usable_pointing_threshold_deg: float = 1.50
    coarse_pointing_threshold_deg: float = 5.00

    # -----------------------------------------------------
    # Slew / settling
    # -----------------------------------------------------
    base_slew_time_s: float = 12.0
    max_slew_time_s: float = 45.0
    slew_time_per_error_deg: float = 2.0

    # -----------------------------------------------------
    # Pointing-error convergence rates (deg/s)
    # Higher = faster reduction in error
    # -----------------------------------------------------
    detumble_error_reduction_dps: float = 0.25
    sun_pointing_error_reduction_dps: float = 0.35
    nadir_pointing_error_reduction_dps: float = 0.40
    ground_tracking_error_reduction_dps: float = 0.38
    inertial_hold_error_reduction_dps: float = 0.18
    safe_sun_acquire_error_reduction_dps: float = 0.28
    momentum_dump_error_penalty_dps: float = -0.05

    # -----------------------------------------------------
    # Body-rate dynamics (deg/s per sec proxy)
    # -----------------------------------------------------
    detumble_rate_reduction_dps2: float = 0.15
    sun_pointing_rate_reduction_dps2: float = 0.06
    nadir_pointing_rate_reduction_dps2: float = 0.08
    ground_tracking_rate_reduction_dps2: float = 0.07
    inertial_hold_rate_reduction_dps2: float = 0.03
    safe_sun_acquire_rate_reduction_dps2: float = 0.08
    slew_rate_excitation_dps2: float = 0.04
    momentum_dump_rate_excitation_dps2: float = 0.03

    # -----------------------------------------------------
    # Wheel momentum dynamics (Nms/s)
    # -----------------------------------------------------
    sun_pointing_momentum_gain_nmsps: float = 0.00025
    nadir_pointing_momentum_gain_nmsps: float = 0.00035
    ground_tracking_momentum_gain_nmsps: float = 0.00040
    inertial_hold_momentum_gain_nmsps: float = 0.00015
    slew_momentum_gain_nmsps: float = 0.00070
    detumble_momentum_gain_nmsps: float = 0.00010
    momentum_dump_release_nmsps: float = 0.00180

    # -----------------------------------------------------
    # Disturbance / passive drift
    # -----------------------------------------------------
    passive_error_growth_dps: float = 0.010
    passive_rate_growth_dps2: float = 0.003
    eclipse_disturbance_multiplier: float = 1.10
    power_stressed_disturbance_multiplier: float = 1.15
    thermal_stressed_disturbance_multiplier: float = 1.10
    sensor_degraded_multiplier: float = 1.35

    # -----------------------------------------------------
    # Lock thresholds
    # -----------------------------------------------------
    lock_threshold_deg: float = 1.2
    precise_lock_threshold_deg: float = 0.4

    # -----------------------------------------------------
    # Numerical guards
    # -----------------------------------------------------
    min_pointing_error_deg: float = 0.0
    max_pointing_error_deg: float = 25.0
    min_body_rate_deg_s: float = 0.0
    max_body_rate_deg_s: float = 15.0
    min_wheel_momentum_nms: float = 0.0
    max_wheel_momentum_nms: float = 0.20


DEFAULT_ADCS_CONFIG = ADCSConfig()


# =========================================================
# Helpers
# =========================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# =========================================================
# Breakdown
# =========================================================

@dataclass(frozen=True)
class ADCSBreakdown:
    """
    Detailed ADCS update report for debugging, testing, and reward shaping.
    """
    attitude_mode: str
    slew_time_remaining_s: float
    pointing_error_deg: float
    body_rate_deg_s: float
    wheel_momentum_nms: float
    nadir_locked: bool
    sun_vector_locked: bool
    ground_track_locked: bool
    wheels_saturated: bool


# =========================================================
# ADCS subsystem
# =========================================================

class ADCSSubsystem:
    """
    Operational ADCS subsystem for a CubeSat simulator.

    Responsibilities:
    - initialize attitude-related state
    - apply mode-aware attitude evolution
    - update pointing error, body rates, and wheel momentum
    - compute lock states
    - manage saturation-related degradation
    """

    def __init__(self, config: ADCSConfig = DEFAULT_ADCS_CONFIG):
        self.config = config

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def initialize(self, state: SpacecraftState) -> None:
        state.adcs.pointing_error_deg = clamp(
            state.adcs.pointing_error_deg if state.adcs.pointing_error_deg > 0 else self.config.initial_pointing_error_deg,
            self.config.min_pointing_error_deg,
            self.config.max_pointing_error_deg,
        )
        state.adcs.body_rate_deg_s = clamp(
            state.adcs.body_rate_deg_s if state.adcs.body_rate_deg_s >= 0 else self.config.initial_body_rate_deg_s,
            self.config.min_body_rate_deg_s,
            self.config.max_body_rate_deg_s,
        )
        state.adcs.wheel_momentum_nms = clamp(
            state.adcs.wheel_momentum_nms if state.adcs.wheel_momentum_nms >= 0 else self.config.initial_wheel_momentum_nms,
            self.config.min_wheel_momentum_nms,
            self.config.max_wheel_momentum_nms,
        )
        state.adcs.wheel_momentum_limit_nms = self.config.wheel_momentum_limit_nms
        self._update_locks(state)

    def step(
        self,
        state: SpacecraftState,
        dt_s: float,
        action: Optional[Action] = None,
        action_profile: Optional[ActionEffectProfile] = None,
    ) -> ADCSBreakdown:
        """
        Advance ADCS state by one simulation step.
        """
        self._maybe_start_slew(state, action)

        disturbance_mult = self._disturbance_multiplier(state)

        # Update slew timer first
        if state.adcs.slew_time_remaining_s > 0.0:
            state.adcs.slew_time_remaining_s = max(0.0, state.adcs.slew_time_remaining_s - dt_s)

        # Update body rate and pointing error according to current attitude mode
        new_body_rate = self._update_body_rate(state, dt_s=dt_s, disturbance_mult=disturbance_mult)
        state.adcs.body_rate_deg_s = clamp(
            new_body_rate,
            self.config.min_body_rate_deg_s,
            self.config.max_body_rate_deg_s,
        )

        new_error = self._update_pointing_error(state, dt_s=dt_s, disturbance_mult=disturbance_mult)
        state.adcs.pointing_error_deg = clamp(
            new_error,
            self.config.min_pointing_error_deg,
            self.config.max_pointing_error_deg,
        )

        new_momentum = self._update_wheel_momentum(state, dt_s=dt_s)
        state.adcs.wheel_momentum_nms = clamp(
            new_momentum,
            self.config.min_wheel_momentum_nms,
            self.config.max_wheel_momentum_nms,
        )

        self._update_locks(state)

        return ADCSBreakdown(
            attitude_mode=state.adcs.mode.name,
            slew_time_remaining_s=state.adcs.slew_time_remaining_s,
            pointing_error_deg=state.adcs.pointing_error_deg,
            body_rate_deg_s=state.adcs.body_rate_deg_s,
            wheel_momentum_nms=state.adcs.wheel_momentum_nms,
            nadir_locked=state.adcs.nadir_locked,
            sun_vector_locked=state.adcs.sun_vector_locked,
            ground_track_locked=state.adcs.ground_track_locked,
            wheels_saturated=state.adcs.wheels_saturated,
        )

    # -----------------------------------------------------
    # Slew / mode transition logic
    # -----------------------------------------------------

    def _maybe_start_slew(self, state: SpacecraftState, action: Optional[Action]) -> None:
        """
        Start a new slew when the commanded action implies a new target attitude.
        """
        if action is None:
            return

        desired_mode = self._desired_attitude_mode_from_action(action)
        if desired_mode is None:
            return

        if desired_mode == state.adcs.mode:
            return

        previous_mode = state.adcs.mode
        state.adcs.mode = desired_mode

        slew_time = self._estimate_slew_time_s(
            current_error_deg=state.adcs.pointing_error_deg,
            previous_mode=previous_mode,
            new_mode=desired_mode,
        )
        state.adcs.slew_time_remaining_s = slew_time

        # Changing major attitude targets causes immediate coarse degradation before settling
        state.adcs.pointing_error_deg = clamp(
            max(state.adcs.pointing_error_deg, 2.0) + 0.8,
            self.config.min_pointing_error_deg,
            self.config.max_pointing_error_deg,
        )

    def _desired_attitude_mode_from_action(self, action: Action) -> Optional[AttitudeMode]:
        mapping = {
            Action.DETUMBLE: AttitudeMode.DETUMBLE,
            Action.SUN_POINT_CHARGE: AttitudeMode.SUN_POINTING,
            Action.NADIR_POINT_STANDBY: AttitudeMode.NADIR_POINTING,
            Action.SLEW_TO_GROUND: AttitudeMode.GROUND_TRACKING,
            Action.HOLD_INERTIAL: AttitudeMode.INERTIAL_HOLD,
            Action.DESATURATE_WHEELS: AttitudeMode.MOMENTUM_DUMP,
            Action.ENTER_SAFE_MODE: AttitudeMode.SAFE_SUN_ACQUIRE,
        }
        return mapping.get(action)

    def _estimate_slew_time_s(
        self,
        current_error_deg: float,
        previous_mode: AttitudeMode,
        new_mode: AttitudeMode,
    ) -> float:
        if previous_mode == new_mode:
            return 0.0

        maneuver_size_deg = max(5.0, current_error_deg * 1.5 + 6.0)
        slew_time = self.config.base_slew_time_s + self.config.slew_time_per_error_deg * maneuver_size_deg
        return clamp(slew_time, self.config.base_slew_time_s, self.config.max_slew_time_s)

    # -----------------------------------------------------
    # Dynamics
    # -----------------------------------------------------

    def _update_body_rate(self, state: SpacecraftState, dt_s: float, disturbance_mult: float) -> float:
        rate = state.adcs.body_rate_deg_s
        mode = state.adcs.mode

        # Baseline passive disturbance
        rate += self.config.passive_rate_growth_dps2 * disturbance_mult * dt_s

        if mode == AttitudeMode.DETUMBLE:
            rate -= self.config.detumble_rate_reduction_dps2 * dt_s
        elif mode == AttitudeMode.SUN_POINTING:
            rate -= self.config.sun_pointing_rate_reduction_dps2 * dt_s
        elif mode == AttitudeMode.NADIR_POINTING:
            rate -= self.config.nadir_pointing_rate_reduction_dps2 * dt_s
        elif mode == AttitudeMode.GROUND_TRACKING:
            rate -= self.config.ground_tracking_rate_reduction_dps2 * dt_s
        elif mode == AttitudeMode.INERTIAL_HOLD:
            rate -= self.config.inertial_hold_rate_reduction_dps2 * dt_s
        elif mode == AttitudeMode.SAFE_SUN_ACQUIRE:
            rate -= self.config.safe_sun_acquire_rate_reduction_dps2 * dt_s
        elif mode == AttitudeMode.SLEW_MANEUVER:
            rate += self.config.slew_rate_excitation_dps2 * dt_s
        elif mode == AttitudeMode.MOMENTUM_DUMP:
            rate += self.config.momentum_dump_rate_excitation_dps2 * dt_s

        # If still slewing, keep rates a bit higher
        if state.adcs.slew_time_remaining_s > 0.0:
            rate += 0.02 * dt_s

        # Saturation degrades fine control
        if state.adcs.wheels_saturated:
            rate += 0.03 * dt_s

        return rate

    def _update_pointing_error(self, state: SpacecraftState, dt_s: float, disturbance_mult: float) -> float:
        err = state.adcs.pointing_error_deg
        mode = state.adcs.mode

        # Passive drift upward
        err += self.config.passive_error_growth_dps * disturbance_mult * dt_s

        # Settling / control authority
        if mode == AttitudeMode.DETUMBLE:
            err -= self.config.detumble_error_reduction_dps * dt_s
        elif mode == AttitudeMode.SUN_POINTING:
            err -= self.config.sun_pointing_error_reduction_dps * dt_s
        elif mode == AttitudeMode.NADIR_POINTING:
            err -= self.config.nadir_pointing_error_reduction_dps * dt_s
        elif mode == AttitudeMode.GROUND_TRACKING:
            err -= self.config.ground_tracking_error_reduction_dps * dt_s
        elif mode == AttitudeMode.INERTIAL_HOLD:
            err -= self.config.inertial_hold_error_reduction_dps * dt_s
        elif mode == AttitudeMode.SAFE_SUN_ACQUIRE:
            err -= self.config.safe_sun_acquire_error_reduction_dps * dt_s
        elif mode == AttitudeMode.MOMENTUM_DUMP:
            err -= self.config.momentum_dump_error_penalty_dps * dt_s  # negative reduction => worsens
        elif mode == AttitudeMode.SLEW_MANEUVER:
            err += 0.08 * dt_s

        # Ongoing slew means not yet settled
        if state.adcs.slew_time_remaining_s > 0.0:
            err += 0.03 * dt_s
        else:
            # Once settled, good modes reduce error slightly faster
            if mode in (
                AttitudeMode.SUN_POINTING,
                AttitudeMode.NADIR_POINTING,
                AttitudeMode.GROUND_TRACKING,
                AttitudeMode.SAFE_SUN_ACQUIRE,
            ):
                err -= 0.03 * dt_s

        # High body rate worsens pointing
        err += 0.04 * state.adcs.body_rate_deg_s * dt_s

        # Wheel saturation hurts precision
        if state.adcs.wheels_saturated:
            err += 0.06 * dt_s

        # Sensor degradation hurts solution quality
        if not self._all_primary_sensors_available(state):
            err += 0.04 * dt_s

        return err

    def _update_wheel_momentum(self, state: SpacecraftState, dt_s: float) -> float:
        momentum = state.adcs.wheel_momentum_nms
        mode = state.adcs.mode

        if mode == AttitudeMode.SUN_POINTING:
            momentum += self.config.sun_pointing_momentum_gain_nmsps * dt_s
        elif mode == AttitudeMode.NADIR_POINTING:
            momentum += self.config.nadir_pointing_momentum_gain_nmsps * dt_s
        elif mode == AttitudeMode.GROUND_TRACKING:
            momentum += self.config.ground_tracking_momentum_gain_nmsps * dt_s
        elif mode == AttitudeMode.INERTIAL_HOLD:
            momentum += self.config.inertial_hold_momentum_gain_nmsps * dt_s
        elif mode == AttitudeMode.SLEW_MANEUVER:
            momentum += self.config.slew_momentum_gain_nmsps * dt_s
        elif mode == AttitudeMode.DETUMBLE:
            momentum += self.config.detumble_momentum_gain_nmsps * dt_s
        elif mode == AttitudeMode.MOMENTUM_DUMP:
            momentum -= self.config.momentum_dump_release_nmsps * dt_s

        # Fast rates tend to build momentum faster
        momentum += 0.00005 * state.adcs.body_rate_deg_s * dt_s

        return momentum

    # -----------------------------------------------------
    # Disturbances / sensor quality
    # -----------------------------------------------------

    def _disturbance_multiplier(self, state: SpacecraftState) -> float:
        mult = 1.0

        if state.orbit.sunlight_state.name == "ECLIPSE":
            mult *= self.config.eclipse_disturbance_multiplier

        if state.eps.mode.name in ("POWER_SAVE", "ECLIPSE_CONSERVE", "CRITICAL_LOW_POWER"):
            mult *= self.config.power_stressed_disturbance_multiplier

        if state.thermal.thermal_violation:
            mult *= self.config.thermal_stressed_disturbance_multiplier

        if not self._all_primary_sensors_available(state):
            mult *= self.config.sensor_degraded_multiplier

        return mult

    def _all_primary_sensors_available(self, state: SpacecraftState) -> bool:
        return (
            state.adcs.imu_available
            and state.adcs.magnetometer_available
            and state.adcs.star_tracker_available
        )

    # -----------------------------------------------------
    # Locks / state flags
    # -----------------------------------------------------

    def _update_locks(self, state: SpacecraftState) -> None:
        err = state.adcs.pointing_error_deg
        settled = state.adcs.slew_time_remaining_s <= 0.0
        mode = state.adcs.mode

        state.adcs.sun_vector_locked = (
            mode in (AttitudeMode.SUN_POINTING, AttitudeMode.SAFE_SUN_ACQUIRE)
            and settled
            and err <= self.config.lock_threshold_deg
        )

        state.adcs.nadir_locked = (
            mode == AttitudeMode.NADIR_POINTING
            and settled
            and err <= self.config.lock_threshold_deg
        )

        state.adcs.ground_track_locked = (
            mode == AttitudeMode.GROUND_TRACKING
            and settled
            and err <= self.config.lock_threshold_deg
        )


# =========================================================
# Functional helpers
# =========================================================

def initialize_adcs_state(
    state: SpacecraftState,
    config: ADCSConfig = DEFAULT_ADCS_CONFIG,
) -> None:
    subsystem = ADCSSubsystem(config=config)
    subsystem.initialize(state)


def update_adcs_state(
    state: SpacecraftState,
    dt_s: float,
    action: Optional[Action] = None,
    action_profile: Optional[ActionEffectProfile] = None,
    config: ADCSConfig = DEFAULT_ADCS_CONFIG,
) -> ADCSBreakdown:
    subsystem = ADCSSubsystem(config=config)
    return subsystem.step(
        state=state,
        dt_s=dt_s,
        action=action,
        action_profile=action_profile,
    )


# =========================================================
# Smoke test helper
# =========================================================

def adcs_smoke_summary(
    state: SpacecraftState,
    dt_s: float = 5.0,
    steps: int = 30,
    action: Optional[Action] = None,
    config: ADCSConfig = DEFAULT_ADCS_CONFIG,
) -> Dict[str, float]:
    """
    Quick ADCS sanity-check helper.
    """
    subsystem = ADCSSubsystem(config=config)

    min_error = state.adcs.pointing_error_deg
    max_error = state.adcs.pointing_error_deg
    min_rate = state.adcs.body_rate_deg_s
    max_rate = state.adcs.body_rate_deg_s
    min_momentum = state.adcs.wheel_momentum_nms
    max_momentum = state.adcs.wheel_momentum_nms

    for i in range(steps):
        breakdown = subsystem.step(
            state=state,
            dt_s=dt_s,
            action=action if i == 0 else None,
        )

        min_error = min(min_error, state.adcs.pointing_error_deg)
        max_error = max(max_error, state.adcs.pointing_error_deg)
        min_rate = min(min_rate, state.adcs.body_rate_deg_s)
        max_rate = max(max_rate, state.adcs.body_rate_deg_s)
        min_momentum = min(min_momentum, state.adcs.wheel_momentum_nms)
        max_momentum = max(max_momentum, state.adcs.wheel_momentum_nms)

    return {
        "final_pointing_error_deg": state.adcs.pointing_error_deg,
        "final_body_rate_deg_s": state.adcs.body_rate_deg_s,
        "final_wheel_momentum_nms": state.adcs.wheel_momentum_nms,
        "final_slew_time_remaining_s": state.adcs.slew_time_remaining_s,
        "min_pointing_error_deg": min_error,
        "max_pointing_error_deg": max_error,
        "min_body_rate_deg_s": min_rate,
        "max_body_rate_deg_s": max_rate,
        "min_wheel_momentum_nms": min_momentum,
        "max_wheel_momentum_nms": max_momentum,
        "sun_vector_locked": float(state.adcs.sun_vector_locked),
        "nadir_locked": float(state.adcs.nadir_locked),
        "ground_track_locked": float(state.adcs.ground_track_locked),
        "wheels_saturated": float(state.adcs.wheels_saturated),
    }