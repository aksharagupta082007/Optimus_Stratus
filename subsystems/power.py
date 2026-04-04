from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from models.action_space import ActionEffectProfile
from models.enums import Action, PowerMode, SunlightState
from models.state_models import EPSState, SpacecraftState


# =========================================================
# Configuration
# =========================================================

@dataclass(frozen=True)
class PowerConfig:
    """
    EPS / power subsystem configuration for a realistic small LEO CubeSat.

    This is an operational power model, not a cell-level battery model.
    It is designed to be:
    - physically sensible
    - numerically stable
    - easy to integrate with RL environments

    Units:
    - Power: W
    - Energy: Wh
    - Voltage: V
    - Current: A
    - Time: s
    - Battery state: %
    """

    # -----------------------------------------------------
    # Battery pack
    # -----------------------------------------------------
    battery_capacity_wh: float = 30.0
    nominal_bus_voltage_v: float = 7.4
    max_bus_voltage_v: float = 8.2
    min_bus_voltage_v: float = 6.4

    battery_min_soc_pct: float = 0.0
    battery_max_soc_pct: float = 100.0

    # Brownout / low-power thresholds
    brownout_soc_threshold_pct: float = 5.0
    critical_soc_threshold_pct: float = 10.0
    low_soc_threshold_pct: float = 30.0
    full_soc_threshold_pct: float = 95.0

    # -----------------------------------------------------
    # Solar generation
    # -----------------------------------------------------
    max_solar_input_w: float = 12.0
    eclipse_solar_input_w: float = 0.0

    # Sun-pointing power bonus factor
    sun_pointing_bonus_factor: float = 1.08

    # Nadir-pointing slight power penalty relative to optimal Sun-pointing
    nadir_pointing_factor: float = 0.88

    # Ground-tracking can be less optimal for solar generation
    ground_tracking_factor: float = 0.82

    # Inertial hold / other generic attitude factor
    inertial_hold_factor: float = 0.90

    # Safe / unknown posture factor
    safe_posture_factor: float = 0.92

    # Beta-angle / geometry coupling strength
    beta_angle_gain: float = 0.10

    # -----------------------------------------------------
    # Base avionics / housekeeping load
    # -----------------------------------------------------
    base_housekeeping_load_w: float = 2.2
    safe_mode_load_w: float = 1.6
    survival_mode_load_w: float = 1.3
    eclipse_power_save_load_w: float = 1.8

    # Subsystem nominal background loads
    adcs_background_load_w: float = 0.7
    payload_background_load_w: float = 0.2
    comms_background_load_w: float = 0.15
    cdh_background_load_w: float = 0.35
    thermal_background_load_w: float = 0.1

    # Heater loads
    battery_heater_load_w: float = 1.2
    payload_heater_load_w: float = 1.5

    # -----------------------------------------------------
    # Mode/load modifiers
    # -----------------------------------------------------
    detumble_extra_load_w: float = 0.9
    momentum_dump_extra_load_w: float = 0.8
    slew_extra_load_w: float = 1.1
    nadir_hold_extra_load_w: float = 0.6
    sun_point_hold_extra_load_w: float = 0.4
    ground_track_hold_extra_load_w: float = 0.9

    payload_warmup_extra_load_w: float = 1.8
    imaging_extra_load_w: float = 2.4
    processing_extra_load_w: float = 1.7

    pass_prep_extra_load_w: float = 0.8
    low_rate_tx_extra_load_w: float = 2.8
    high_rate_tx_extra_load_w: float = 4.1
    beacon_extra_load_w: float = 1.1

    buffering_extra_load_w: float = 0.25
    compressing_extra_load_w: float = 0.7
    downlinking_cdh_extra_load_w: float = 0.3
    recovery_extra_load_w: float = 0.4

    # -----------------------------------------------------
    # Battery efficiency / charging-discharge behavior
    # -----------------------------------------------------
    charge_efficiency: float = 0.93
    discharge_efficiency: float = 0.95

    # SoC-dependent charging taper near full battery
    taper_start_soc_pct: float = 85.0
    taper_end_soc_pct: float = 100.0
    taper_min_factor: float = 0.15

    # Temperature derating
    cold_charge_temp_c: float = 0.0
    hot_charge_temp_c: float = 40.0
    cold_discharge_temp_c: float = -5.0
    hot_discharge_temp_c: float = 45.0

    severe_temp_derate_factor: float = 0.45
    mild_temp_derate_factor: float = 0.75

    # -----------------------------------------------------
    # Numerical guards
    # -----------------------------------------------------
    min_load_w: float = 0.0
    max_load_w: float = 30.0
    min_solar_w: float = 0.0
    max_solar_w: float = 25.0


DEFAULT_POWER_CONFIG = PowerConfig()


# =========================================================
# Helpers
# =========================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t


# =========================================================
# Load breakdown
# =========================================================

@dataclass(frozen=True)
class PowerBreakdown:
    """
    Detailed power accounting for debugging and reward shaping.
    """
    solar_input_w: float
    base_load_w: float
    adcs_load_w: float
    payload_load_w: float
    comms_load_w: float
    cdh_load_w: float
    thermal_load_w: float
    action_extra_load_w: float
    total_load_w: float
    net_power_w: float
    battery_current_a: float
    battery_voltage_v: float
    delta_soc_pct: float
    power_positive: bool


# =========================================================
# Power subsystem model
# =========================================================

class PowerSubsystem:
    """
    Operational EPS model for the spacecraft.

    Responsibilities:
    - compute solar input based on sunlight + posture
    - compute total electrical load from spacecraft/subsystem state
    - update battery SoC, bus voltage, current, and power flags
    - expose interpretable power breakdowns for testing/debugging
    """

    def __init__(self, config: PowerConfig = DEFAULT_POWER_CONFIG):
        self.config = config

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def initialize(self, eps: EPSState) -> EPSState:
        eps.battery_capacity_wh = self.config.battery_capacity_wh
        eps.min_soc_pct = self.config.battery_min_soc_pct
        eps.max_soc_pct = self.config.battery_max_soc_pct
        eps.battery_soc_pct = clamp(
            eps.battery_soc_pct,
            self.config.battery_min_soc_pct,
            self.config.battery_max_soc_pct,
        )
        eps.battery_voltage_v = self._estimate_bus_voltage(eps.battery_soc_pct)
        eps.update_net_power()
        return eps

    def step(
        self,
        state: SpacecraftState,
        dt_s: float,
        action: Optional[Action] = None,
        action_profile: Optional[ActionEffectProfile] = None,
    ) -> PowerBreakdown:
        """
        Update EPS state by one simulation step.

        The environment can provide:
        - action: high-level action enum
        - action_profile: precomputed action effect from action_space.py

        If both are provided, action_profile.extra_load_w is used directly
        as the action-induced incremental load term.
        """
        solar_input_w = self.compute_solar_input_w(state)
        total_load_w, component_loads = self.compute_total_load_w(
            state=state,
            action=action,
            action_profile=action_profile,
        )

        net_power_w = solar_input_w - total_load_w
        delta_soc_pct = self._compute_delta_soc_pct(
            battery_soc_pct=state.eps.battery_soc_pct,
            battery_temp_c=state.thermal.battery_temp_c,
            net_power_w=net_power_w,
            dt_s=dt_s,
        )

        new_soc = clamp(
            state.eps.battery_soc_pct + delta_soc_pct,
            self.config.battery_min_soc_pct,
            self.config.battery_max_soc_pct,
        )

        state.eps.solar_input_w = solar_input_w
        state.eps.load_output_w = total_load_w
        state.eps.net_power_w = net_power_w
        state.eps.battery_soc_pct = new_soc
        state.eps.battery_voltage_v = self._estimate_bus_voltage(new_soc)
        state.eps.battery_current_a = self._estimate_battery_current_a(
            net_power_w=net_power_w,
            bus_voltage_v=state.eps.battery_voltage_v,
        )
        state.eps.battery_temp_c = state.thermal.battery_temp_c
        state.eps.eps_power_positive = net_power_w >= 0.0
        state.eps.brownout_risk = new_soc <= self.config.brownout_soc_threshold_pct

        self._update_power_mode(state)

        return PowerBreakdown(
            solar_input_w=solar_input_w,
            base_load_w=component_loads["base"],
            adcs_load_w=component_loads["adcs"],
            payload_load_w=component_loads["payload"],
            comms_load_w=component_loads["comms"],
            cdh_load_w=component_loads["cdh"],
            thermal_load_w=component_loads["thermal"],
            action_extra_load_w=component_loads["action"],
            total_load_w=total_load_w,
            net_power_w=net_power_w,
            battery_current_a=state.eps.battery_current_a,
            battery_voltage_v=state.eps.battery_voltage_v,
            delta_soc_pct=delta_soc_pct,
            power_positive=state.eps.eps_power_positive,
        )

    def compute_solar_input_w(self, state: SpacecraftState) -> float:
        """
        Estimate solar input based on orbit lighting and spacecraft posture.
        """
        if state.orbit.sunlight_state == SunlightState.ECLIPSE:
            return self.config.eclipse_solar_input_w

        sun_factor = self._sun_geometry_factor(
            sun_elevation_deg=state.orbit.sun_elevation_deg,
            beta_angle_deg=state.orbit.beta_angle_deg,
        )
        posture_factor = self._posture_generation_factor(state)
        soc_taper = self._charge_taper_factor(state.eps.battery_soc_pct)
        temp_derate = self._charge_temperature_factor(state.thermal.battery_temp_c)

        solar_w = (
            self.config.max_solar_input_w
            * sun_factor
            * posture_factor
            * soc_taper
            * temp_derate
        )
        return clamp(solar_w, self.config.min_solar_w, self.config.max_solar_w)

    def compute_total_load_w(
        self,
        state: SpacecraftState,
        action: Optional[Action] = None,
        action_profile: Optional[ActionEffectProfile] = None,
    ) -> tuple[float, Dict[str, float]]:
        """
        Compute electrical load from all subsystems and the current action.
        """
        base_load = self._base_load_for_spacecraft_mode(state)
        adcs_load = self._adcs_load_w(state)
        payload_load = self._payload_load_w(state)
        comms_load = self._comms_load_w(state)
        cdh_load = self._cdh_load_w(state)
        thermal_load = self._thermal_load_w(state)
        action_extra = self._action_extra_load_w(state, action=action, action_profile=action_profile)

        total = (
            base_load
            + adcs_load
            + payload_load
            + comms_load
            + cdh_load
            + thermal_load
            + action_extra
        )
        total = clamp(total, self.config.min_load_w, self.config.max_load_w)

        return total, {
            "base": base_load,
            "adcs": adcs_load,
            "payload": payload_load,
            "comms": comms_load,
            "cdh": cdh_load,
            "thermal": thermal_load,
            "action": action_extra,
        }

    # -----------------------------------------------------
    # Solar generation model
    # -----------------------------------------------------

    def _sun_geometry_factor(self, sun_elevation_deg: float, beta_angle_deg: float) -> float:
        """
        Simple operational solar generation proxy.

        Uses:
        - sun elevation over the scene/orbit proxy
        - beta-angle-like factor for long-term geometry modulation
        """
        # Normalize sun elevation from [0, 90]
        elev_factor = clamp(sun_elevation_deg / 90.0, 0.0, 1.0)

        # Beta-angle modifier around ~15-25 deg typical proxy range
        beta_norm = clamp(abs(beta_angle_deg) / 45.0, 0.0, 1.0)
        beta_factor = 1.0 - self.config.beta_angle_gain + self.config.beta_angle_gain * beta_norm

        return clamp(elev_factor * beta_factor, 0.0, 1.0)

    def _posture_generation_factor(self, state: SpacecraftState) -> float:
        """
        How favorable the current spacecraft posture is for solar collection.
        """
        mode = state.adcs.mode.name

        if mode == "SUN_POINTING" or state.eps.mode == PowerMode.CHARGE_PRIORITY:
            return self.config.sun_pointing_bonus_factor
        if mode == "NADIR_POINTING":
            return self.config.nadir_pointing_factor
        if mode == "GROUND_TRACKING":
            return self.config.ground_tracking_factor
        if mode == "INERTIAL_HOLD":
            return self.config.inertial_hold_factor
        return self.config.safe_posture_factor

    def _charge_taper_factor(self, soc_pct: float) -> float:
        """
        Reduce effective charge power as battery approaches full.
        """
        if soc_pct <= self.config.taper_start_soc_pct:
            return 1.0
        if soc_pct >= self.config.taper_end_soc_pct:
            return self.config.taper_min_factor

        t = (
            (soc_pct - self.config.taper_start_soc_pct)
            / max(self.config.taper_end_soc_pct - self.config.taper_start_soc_pct, 1e-9)
        )
        return lerp(1.0, self.config.taper_min_factor, t)

    def _charge_temperature_factor(self, battery_temp_c: float) -> float:
        """
        Charging is less efficient / more constrained outside nominal temperature.
        """
        if battery_temp_c < self.config.cold_charge_temp_c - 5.0:
            return self.config.severe_temp_derate_factor
        if battery_temp_c < self.config.cold_charge_temp_c:
            return self.config.mild_temp_derate_factor
        if battery_temp_c > self.config.hot_charge_temp_c + 5.0:
            return self.config.severe_temp_derate_factor
        if battery_temp_c > self.config.hot_charge_temp_c:
            return self.config.mild_temp_derate_factor
        return 1.0

    # -----------------------------------------------------
    # Electrical load model
    # -----------------------------------------------------

    def _base_load_for_spacecraft_mode(self, state: SpacecraftState) -> float:
        if state.mode == state.mode.SAFE:
            return self.config.safe_mode_load_w
        if state.mode == state.mode.SURVIVAL:
            return self.config.survival_mode_load_w
        if state.mode == state.mode.ECLIPSE_POWER_SAVE:
            return self.config.eclipse_power_save_load_w
        return self.config.base_housekeeping_load_w

    def _adcs_load_w(self, state: SpacecraftState) -> float:
        load = self.config.adcs_background_load_w
        mode = state.adcs.mode.name

        if mode == "DETUMBLE":
            load += self.config.detumble_extra_load_w
        elif mode == "MOMENTUM_DUMP":
            load += self.config.momentum_dump_extra_load_w
        elif mode == "SLEW_MANEUVER":
            load += self.config.slew_extra_load_w
        elif mode == "NADIR_POINTING":
            load += self.config.nadir_hold_extra_load_w
        elif mode == "SUN_POINTING":
            load += self.config.sun_point_hold_extra_load_w
        elif mode == "GROUND_TRACKING":
            load += self.config.ground_track_hold_extra_load_w

        # Higher body rates and larger pointing errors cost more control effort.
        rate_factor = clamp(state.adcs.body_rate_deg_s / 5.0, 0.0, 1.5)
        err_factor = clamp(state.adcs.pointing_error_deg / 10.0, 0.0, 1.0)
        load += 0.35 * rate_factor + 0.25 * err_factor

        # Wheel saturation often implies more control effort.
        if state.adcs.wheels_saturated:
            load += 0.25

        return load

    def _payload_load_w(self, state: SpacecraftState) -> float:
        load = self.config.payload_background_load_w
        mode = state.payload.mode.name

        if mode == "WARMUP":
            load += self.config.payload_warmup_extra_load_w
        elif mode == "IMAGING":
            load += self.config.imaging_extra_load_w
        elif mode == "PROCESSING":
            load += self.config.processing_extra_load_w
        elif mode == "READY":
            load += 0.35

        if state.payload.payload_enabled:
            load += 0.15

        return load

    def _comms_load_w(self, state: SpacecraftState) -> float:
        load = self.config.comms_background_load_w
        mode = state.comms.mode.name

        if mode == "PASS_PREP":
            load += self.config.pass_prep_extra_load_w
        elif mode == "LOW_RATE_TX":
            load += self.config.low_rate_tx_extra_load_w
        elif mode == "HIGH_RATE_TX":
            load += self.config.high_rate_tx_extra_load_w
        elif mode == "BEACON":
            load += self.config.beacon_extra_load_w
        elif mode in ("LISTEN", "RX_ONLY", "TXRX"):
            load += 0.25

        # Poor links can cost slightly more due to retries / inefficiency
        if state.comms.gs_visible and state.comms.link_quality.name in ("POOR", "MARGINAL"):
            load += 0.15

        return load

    def _cdh_load_w(self, state: SpacecraftState) -> float:
        load = self.config.cdh_background_load_w
        mode = state.cdh.mode.name

        if mode == "BUFFERING":
            load += self.config.buffering_extra_load_w
        elif mode == "CLASSIFYING":
            load += self.config.processing_extra_load_w
        elif mode == "COMPRESSING":
            load += self.config.compressing_extra_load_w
        elif mode == "DOWNLINKING":
            load += self.config.downlinking_cdh_extra_load_w
        elif mode == "RECOVERY":
            load += self.config.recovery_extra_load_w

        # Slight load growth with memory pressure / CPU usage
        load += 0.002 * clamp(state.cdh.cpu_load_pct, 0.0, 100.0)

        return load

    def _thermal_load_w(self, state: SpacecraftState) -> float:
        load = self.config.thermal_background_load_w

        if state.thermal.battery_heater_on:
            load += self.config.battery_heater_load_w
        if state.thermal.payload_heater_on:
            load += self.config.payload_heater_load_w

        return load

    def _action_extra_load_w(
        self,
        state: SpacecraftState,
        action: Optional[Action],
        action_profile: Optional[ActionEffectProfile],
    ) -> float:
        """
        Optional transient action-specific load. This lets the environment
        model short-lived step actions on top of subsystem modes.
        """
        if action_profile is not None:
            return max(0.0, action_profile.extra_load_w)

        if action is None:
            return 0.0

        # Conservative fallback if profile is not supplied
        name = action.name
        if name == "CAPTURE_IMAGE":
            return 1.0
        if name == "RUN_CLASSIFIER":
            return 0.8
        if name == "STORE_FRAME":
            return 0.2
        if name == "DISCARD_FRAME":
            return 0.05
        if name == "PREPARE_DOWNLINK":
            return 0.4
        if name == "DOWNLINK_LOW_RATE":
            return 0.8
        if name == "DOWNLINK_HIGH_RATE":
            return 1.0
        if name == "DESATURATE_WHEELS":
            return 0.4
        return 0.0

    # -----------------------------------------------------
    # Battery state update
    # -----------------------------------------------------

    def _compute_delta_soc_pct(
        self,
        battery_soc_pct: float,
        battery_temp_c: float,
        net_power_w: float,
        dt_s: float,
    ) -> float:
        """
        Convert net power over dt into battery SoC change.

        Positive net power charges battery.
        Negative net power discharges battery.
        """
        dt_h = dt_s / 3600.0
        if dt_h <= 0.0 or self.config.battery_capacity_wh <= 0.0:
            return 0.0

        if net_power_w >= 0.0:
            effective_power_w = net_power_w * self.config.charge_efficiency * self._charge_temperature_factor(battery_temp_c)
            delta_wh = effective_power_w * dt_h
            delta_soc = 100.0 * delta_wh / self.config.battery_capacity_wh
            return delta_soc

        discharge_factor = self._discharge_temperature_factor(battery_temp_c)
        effective_draw_w = abs(net_power_w) / max(self.config.discharge_efficiency * discharge_factor, 1e-9)
        delta_wh = effective_draw_w * dt_h
        delta_soc = -100.0 * delta_wh / self.config.battery_capacity_wh
        return delta_soc

    def _discharge_temperature_factor(self, battery_temp_c: float) -> float:
        """
        Discharge performance degrades in severe cold/hot conditions.
        """
        if battery_temp_c < self.config.cold_discharge_temp_c - 5.0:
            return self.config.severe_temp_derate_factor
        if battery_temp_c < self.config.cold_discharge_temp_c:
            return self.config.mild_temp_derate_factor
        if battery_temp_c > self.config.hot_discharge_temp_c + 5.0:
            return self.config.severe_temp_derate_factor
        if battery_temp_c > self.config.hot_discharge_temp_c:
            return self.config.mild_temp_derate_factor
        return 1.0

    def _estimate_bus_voltage(self, soc_pct: float) -> float:
        """
        Smooth bus-voltage proxy from SoC.
        """
        soc_norm = clamp(soc_pct / 100.0, 0.0, 1.0)
        # Slightly convex curve for a more realistic Li-ion-like voltage shape
        shaped = soc_norm ** 0.55
        return lerp(self.config.min_bus_voltage_v, self.config.max_bus_voltage_v, shaped)

    def _estimate_battery_current_a(self, net_power_w: float, bus_voltage_v: float) -> float:
        """
        Positive current means charging into battery; negative means discharge.
        """
        bus_voltage_v = max(bus_voltage_v, 1e-6)
        return net_power_w / bus_voltage_v

    # -----------------------------------------------------
    # EPS operating policy
    # -----------------------------------------------------

    def _update_power_mode(self, state: SpacecraftState) -> None:
        soc = state.eps.battery_soc_pct

        if soc <= self.config.critical_soc_threshold_pct:
            state.eps.mode = PowerMode.CRITICAL_LOW_POWER
            return

        if state.orbit.sunlight_state == SunlightState.ECLIPSE and soc <= self.config.low_soc_threshold_pct:
            state.eps.mode = PowerMode.ECLIPSE_CONSERVE
            return

        if soc <= self.config.low_soc_threshold_pct and state.orbit.sunlight_state == SunlightState.SUNLIT:
            state.eps.mode = PowerMode.CHARGE_PRIORITY
            return

        if state.eps.net_power_w < 0.0:
            state.eps.mode = PowerMode.POWER_SAVE
            return

        if soc >= self.config.full_soc_threshold_pct and state.eps.net_power_w >= 0.0:
            state.eps.mode = PowerMode.POWER_POSITIVE
            return

        state.eps.mode = PowerMode.NOMINAL


# =========================================================
# Functional helpers
# =========================================================

def initialize_eps(
    eps: EPSState,
    config: PowerConfig = DEFAULT_POWER_CONFIG,
) -> EPSState:
    subsystem = PowerSubsystem(config=config)
    return subsystem.initialize(eps)


def update_power_state(
    state: SpacecraftState,
    dt_s: float,
    action: Optional[Action] = None,
    action_profile: Optional[ActionEffectProfile] = None,
    config: PowerConfig = DEFAULT_POWER_CONFIG,
) -> PowerBreakdown:
    subsystem = PowerSubsystem(config=config)
    return subsystem.step(
        state=state,
        dt_s=dt_s,
        action=action,
        action_profile=action_profile,
    )


# =========================================================
# Smoke test helper
# =========================================================

def power_smoke_summary(
    state: SpacecraftState,
    dt_s: float = 5.0,
    steps: int = 20,
    config: PowerConfig = DEFAULT_POWER_CONFIG,
) -> Dict[str, float]:
    """
    Quick EPS sanity check utility.
    Runs repeated power updates using the current spacecraft modes/state.
    """
    subsystem = PowerSubsystem(config=config)

    initial_soc = state.eps.battery_soc_pct
    min_soc = initial_soc
    max_soc = initial_soc
    min_net = 1e9
    max_net = -1e9

    for _ in range(steps):
        breakdown = subsystem.step(state=state, dt_s=dt_s)
        min_soc = min(min_soc, state.eps.battery_soc_pct)
        max_soc = max(max_soc, state.eps.battery_soc_pct)
        min_net = min(min_net, breakdown.net_power_w)
        max_net = max(max_net, breakdown.net_power_w)

    return {
        "initial_soc_pct": initial_soc,
        "final_soc_pct": state.eps.battery_soc_pct,
        "min_soc_pct": min_soc,
        "max_soc_pct": max_soc,
        "final_solar_input_w": state.eps.solar_input_w,
        "final_load_output_w": state.eps.load_output_w,
        "final_net_power_w": state.eps.net_power_w,
        "min_net_power_w": min_net,
        "max_net_power_w": max_net,
        "battery_voltage_v": state.eps.battery_voltage_v,
        "battery_current_a": state.eps.battery_current_a,
    }