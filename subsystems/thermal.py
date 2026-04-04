from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from models.action_space import ActionEffectProfile
from models.enums import Action, ThermalMode, SunlightState
from models.state_models import SpacecraftState


# =========================================================
# Configuration
# =========================================================

@dataclass(frozen=True)
class ThermalConfig:
    """
    Operational thermal model for a small Earth-observation CubeSat.

    This is not a full multi-node thermal solver. It is a stable mission-simulator
    model that captures the most important operational effects:

    - sunlit vs eclipse heating/cooling
    - battery, payload, radio, and bus temperature evolution
    - heater influence
    - activity-driven internal heat generation
    - weak thermal coupling between nodes
    - automatic thermal mode selection

    Units:
    - Temperature: deg C
    - Time: s
    - Power-like heating terms are represented as temperature-rate effects
      to keep the simulator lightweight and numerically stable.
    """

    # -----------------------------------------------------
    # Environmental reference temperatures
    # -----------------------------------------------------
    sunlit_equilibrium_bus_c: float = 24.0
    eclipse_equilibrium_bus_c: float = -8.0

    sunlit_equilibrium_battery_c: float = 16.0
    eclipse_equilibrium_battery_c: float = 2.0

    sunlit_equilibrium_payload_c: float = 18.0
    eclipse_equilibrium_payload_c: float = -2.0

    sunlit_equilibrium_radio_c: float = 20.0
    eclipse_equilibrium_radio_c: float = -4.0

    # -----------------------------------------------------
    # Passive thermal time constants (seconds)
    # Larger => slower temperature response
    # -----------------------------------------------------
    bus_time_constant_s: float = 900.0
    battery_time_constant_s: float = 1400.0
    payload_time_constant_s: float = 1000.0
    radio_time_constant_s: float = 700.0

    # -----------------------------------------------------
    # Internal heating rates (deg C / sec at full activity)
    # These are tuned operational proxies, not physical material constants.
    # -----------------------------------------------------
    battery_charge_heating_rate_cps: float = 0.0025
    battery_discharge_heating_rate_cps: float = 0.0035

    payload_ready_heating_rate_cps: float = 0.0012
    payload_warmup_heating_rate_cps: float = 0.0080
    payload_imaging_heating_rate_cps: float = 0.0060
    payload_processing_heating_rate_cps: float = 0.0050

    radio_listen_heating_rate_cps: float = 0.0012
    radio_beacon_heating_rate_cps: float = 0.0030
    radio_low_rate_tx_heating_rate_cps: float = 0.0060
    radio_high_rate_tx_heating_rate_cps: float = 0.0110

    adcs_hold_bus_heating_rate_cps: float = 0.0015
    adcs_slew_bus_heating_rate_cps: float = 0.0045
    adcs_detumble_bus_heating_rate_cps: float = 0.0035
    adcs_momentum_dump_bus_heating_rate_cps: float = 0.0030

    cdh_background_heating_rate_cps: float = 0.0010
    cdh_processing_heating_rate_cps: float = 0.0025
    cdh_recovery_heating_rate_cps: float = 0.0030

    # -----------------------------------------------------
    # Heater effects (deg C / sec)
    # -----------------------------------------------------
    battery_heater_rate_cps: float = 0.0120
    payload_heater_rate_cps: float = 0.0100

    # -----------------------------------------------------
    # Cross-coupling factors between thermal nodes
    # Values are small by design to avoid instability.
    # -----------------------------------------------------
    bus_to_battery_coupling: float = 0.08
    bus_to_payload_coupling: float = 0.10
    bus_to_radio_coupling: float = 0.07

    payload_to_bus_coupling: float = 0.05
    radio_to_bus_coupling: float = 0.04
    battery_to_bus_coupling: float = 0.03

    # -----------------------------------------------------
    # Net-power influence
    # Positive charging and negative heavy discharge both add some battery heat.
    # -----------------------------------------------------
    charge_heat_gain_per_w: float = 0.0008
    discharge_heat_gain_per_w: float = 0.0011

    # -----------------------------------------------------
    # Safety / control thresholds
    # -----------------------------------------------------
    battery_heat_enable_below_c: float = 4.0
    battery_heat_disable_above_c: float = 8.0

    payload_heat_enable_below_c: float = 0.0
    payload_heat_disable_above_c: float = 6.0

    cold_soak_threshold_c: float = 0.0
    hot_soak_threshold_c: float = 38.0

    # -----------------------------------------------------
    # Numerical protection
    # -----------------------------------------------------
    min_temp_c: float = -80.0
    max_temp_c: float = 90.0


DEFAULT_THERMAL_CONFIG = ThermalConfig()


# =========================================================
# Helpers
# =========================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def first_order_relaxation(current: float, equilibrium: float, dt_s: float, tau_s: float) -> float:
    """
    Stable first-order thermal relaxation step:
        dT = (Teq - T) * dt / tau
    """
    tau_s = max(tau_s, 1e-6)
    return (equilibrium - current) * (dt_s / tau_s)


# =========================================================
# Breakdown
# =========================================================

@dataclass(frozen=True)
class ThermalBreakdown:
    """
    Detailed thermal accounting for logging and validation.
    """
    env_bus_eq_c: float
    env_battery_eq_c: float
    env_payload_eq_c: float
    env_radio_eq_c: float

    bus_delta_c: float
    battery_delta_c: float
    payload_delta_c: float
    radio_delta_c: float

    battery_heater_on: bool
    payload_heater_on: bool

    bus_temp_c: float
    battery_temp_c: float
    payload_temp_c: float
    radio_temp_c: float


# =========================================================
# Thermal subsystem
# =========================================================

class ThermalSubsystem:
    """
    Operational spacecraft thermal model.

    Responsibilities:
    - update bus, battery, payload, and radio temperatures
    - model environmental heating/cooling from sunlight/eclipse
    - model activity-driven heating from subsystems
    - model heater effects
    - update thermal mode and keep temperatures bounded
    """

    def __init__(self, config: ThermalConfig = DEFAULT_THERMAL_CONFIG):
        self.config = config

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def initialize(self, state: SpacecraftState) -> None:
        state.thermal.bus_temp_c = clamp(state.thermal.bus_temp_c, self.config.min_temp_c, self.config.max_temp_c)
        state.thermal.battery_temp_c = clamp(state.thermal.battery_temp_c, self.config.min_temp_c, self.config.max_temp_c)
        state.thermal.payload_temp_c = clamp(state.thermal.payload_temp_c, self.config.min_temp_c, self.config.max_temp_c)
        state.thermal.radio_temp_c = clamp(state.thermal.radio_temp_c, self.config.min_temp_c, self.config.max_temp_c)
        self._update_thermal_mode(state)

    def step(
        self,
        state: SpacecraftState,
        dt_s: float,
        action: Optional[Action] = None,
        action_profile: Optional[ActionEffectProfile] = None,
    ) -> ThermalBreakdown:
        """
        Advance thermal state by one simulation step.
        """
        env_eq = self._environment_equilibria(state)

        bus_delta = self._compute_bus_delta_c(
            state=state,
            dt_s=dt_s,
            env_eq_c=env_eq["bus"],
            action=action,
            action_profile=action_profile,
        )

        battery_delta = self._compute_battery_delta_c(
            state=state,
            dt_s=dt_s,
            env_eq_c=env_eq["battery"],
        )

        payload_delta = self._compute_payload_delta_c(
            state=state,
            dt_s=dt_s,
            env_eq_c=env_eq["payload"],
            action=action,
            action_profile=action_profile,
        )

        radio_delta = self._compute_radio_delta_c(
            state=state,
            dt_s=dt_s,
            env_eq_c=env_eq["radio"],
            action=action,
            action_profile=action_profile,
        )

        # Apply updates
        state.thermal.bus_temp_c = clamp(
            state.thermal.bus_temp_c + bus_delta,
            self.config.min_temp_c,
            self.config.max_temp_c,
        )
        state.thermal.battery_temp_c = clamp(
            state.thermal.battery_temp_c + battery_delta,
            self.config.min_temp_c,
            self.config.max_temp_c,
        )
        state.thermal.payload_temp_c = clamp(
            state.thermal.payload_temp_c + payload_delta,
            self.config.min_temp_c,
            self.config.max_temp_c,
        )
        state.thermal.radio_temp_c = clamp(
            state.thermal.radio_temp_c + radio_delta,
            self.config.min_temp_c,
            self.config.max_temp_c,
        )

        # Keep EPS battery temperature mirrored
        state.eps.battery_temp_c = state.thermal.battery_temp_c

        self._update_thermal_mode(state)

        return ThermalBreakdown(
            env_bus_eq_c=env_eq["bus"],
            env_battery_eq_c=env_eq["battery"],
            env_payload_eq_c=env_eq["payload"],
            env_radio_eq_c=env_eq["radio"],
            bus_delta_c=bus_delta,
            battery_delta_c=battery_delta,
            payload_delta_c=payload_delta,
            radio_delta_c=radio_delta,
            battery_heater_on=state.thermal.battery_heater_on,
            payload_heater_on=state.thermal.payload_heater_on,
            bus_temp_c=state.thermal.bus_temp_c,
            battery_temp_c=state.thermal.battery_temp_c,
            payload_temp_c=state.thermal.payload_temp_c,
            radio_temp_c=state.thermal.radio_temp_c,
        )

    # -----------------------------------------------------
    # Environment equilibria
    # -----------------------------------------------------

    def _environment_equilibria(self, state: SpacecraftState) -> Dict[str, float]:
        sunlit = state.orbit.sunlight_state == SunlightState.SUNLIT

        if sunlit:
            return {
                "bus": self.config.sunlit_equilibrium_bus_c,
                "battery": self.config.sunlit_equilibrium_battery_c,
                "payload": self.config.sunlit_equilibrium_payload_c,
                "radio": self.config.sunlit_equilibrium_radio_c,
            }

        return {
            "bus": self.config.eclipse_equilibrium_bus_c,
            "battery": self.config.eclipse_equilibrium_battery_c,
            "payload": self.config.eclipse_equilibrium_payload_c,
            "radio": self.config.eclipse_equilibrium_radio_c,
        }

    # -----------------------------------------------------
    # Node update equations
    # -----------------------------------------------------

    def _compute_bus_delta_c(
        self,
        state: SpacecraftState,
        dt_s: float,
        env_eq_c: float,
        action: Optional[Action],
        action_profile: Optional[ActionEffectProfile],
    ) -> float:
        current = state.thermal.bus_temp_c

        # Relaxation toward environment
        delta = first_order_relaxation(
            current=current,
            equilibrium=env_eq_c,
            dt_s=dt_s,
            tau_s=self.config.bus_time_constant_s,
        )

        # ADCS-driven internal heat
        adcs_mode = state.adcs.mode.name
        if adcs_mode in ("SUN_POINTING", "NADIR_POINTING", "GROUND_TRACKING", "INERTIAL_HOLD", "SAFE_SUN_ACQUIRE"):
            delta += self.config.adcs_hold_bus_heating_rate_cps * dt_s
        elif adcs_mode == "SLEW_MANEUVER":
            delta += self.config.adcs_slew_bus_heating_rate_cps * dt_s
        elif adcs_mode == "DETUMBLE":
            delta += self.config.adcs_detumble_bus_heating_rate_cps * dt_s
        elif adcs_mode == "MOMENTUM_DUMP":
            delta += self.config.adcs_momentum_dump_bus_heating_rate_cps * dt_s

        # CDH activity
        cdh_mode = state.cdh.mode.name
        delta += self.config.cdh_background_heating_rate_cps * dt_s
        if cdh_mode in ("CLASSIFYING", "COMPRESSING", "DOWNLINKING"):
            delta += self.config.cdh_processing_heating_rate_cps * dt_s
        elif cdh_mode == "RECOVERY":
            delta += self.config.cdh_recovery_heating_rate_cps * dt_s

        # Coupling from other nodes
        delta += self.config.payload_to_bus_coupling * (state.thermal.payload_temp_c - current) * (dt_s / self.config.bus_time_constant_s)
        delta += self.config.radio_to_bus_coupling * (state.thermal.radio_temp_c - current) * (dt_s / self.config.bus_time_constant_s)
        delta += self.config.battery_to_bus_coupling * (state.thermal.battery_temp_c - current) * (dt_s / self.config.bus_time_constant_s)

        # Transient action heat contribution
        if action_profile is not None:
            delta += 0.15 * action_profile.heat_delta_c

        return delta

    def _compute_battery_delta_c(
        self,
        state: SpacecraftState,
        dt_s: float,
        env_eq_c: float,
    ) -> float:
        current = state.thermal.battery_temp_c

        delta = first_order_relaxation(
            current=current,
            equilibrium=env_eq_c,
            dt_s=dt_s,
            tau_s=self.config.battery_time_constant_s,
        )

        # Coupling with bus
        delta += self.config.bus_to_battery_coupling * (state.thermal.bus_temp_c - current) * (dt_s / self.config.battery_time_constant_s)

        # Battery charging/discharging heat from EPS net power
        net_power = state.eps.net_power_w
        if net_power >= 0.0:
            delta += net_power * self.config.charge_heat_gain_per_w * dt_s
            delta += self.config.battery_charge_heating_rate_cps * dt_s
        else:
            delta += abs(net_power) * self.config.discharge_heat_gain_per_w * dt_s
            delta += self.config.battery_discharge_heating_rate_cps * dt_s

        # Heater
        if state.thermal.battery_heater_on:
            delta += self.config.battery_heater_rate_cps * dt_s

        return delta

    def _compute_payload_delta_c(
        self,
        state: SpacecraftState,
        dt_s: float,
        env_eq_c: float,
        action: Optional[Action],
        action_profile: Optional[ActionEffectProfile],
    ) -> float:
        current = state.thermal.payload_temp_c

        delta = first_order_relaxation(
            current=current,
            equilibrium=env_eq_c,
            dt_s=dt_s,
            tau_s=self.config.payload_time_constant_s,
        )

        # Coupling with bus
        delta += self.config.bus_to_payload_coupling * (state.thermal.bus_temp_c - current) * (dt_s / self.config.payload_time_constant_s)

        # Payload operational heat
        payload_mode = state.payload.mode.name
        if payload_mode == "READY":
            delta += self.config.payload_ready_heating_rate_cps * dt_s
        elif payload_mode == "WARMUP":
            delta += self.config.payload_warmup_heating_rate_cps * dt_s
        elif payload_mode == "IMAGING":
            delta += self.config.payload_imaging_heating_rate_cps * dt_s
        elif payload_mode == "PROCESSING":
            delta += self.config.payload_processing_heating_rate_cps * dt_s

        # Heater
        if state.thermal.payload_heater_on:
            delta += self.config.payload_heater_rate_cps * dt_s

        # Transient action heat contribution
        if action_profile is not None and action_profile.payload_use:
            delta += 0.35 * action_profile.heat_delta_c

        return delta

    def _compute_radio_delta_c(
        self,
        state: SpacecraftState,
        dt_s: float,
        env_eq_c: float,
        action: Optional[Action],
        action_profile: Optional[ActionEffectProfile],
    ) -> float:
        current = state.thermal.radio_temp_c

        delta = first_order_relaxation(
            current=current,
            equilibrium=env_eq_c,
            dt_s=dt_s,
            tau_s=self.config.radio_time_constant_s,
        )

        # Coupling with bus
        delta += self.config.bus_to_radio_coupling * (state.thermal.bus_temp_c - current) * (dt_s / self.config.radio_time_constant_s)

        # Comms activity heat
        comms_mode = state.comms.mode.name
        if comms_mode in ("LISTEN", "RX_ONLY", "TXRX", "PASS_PREP"):
            delta += self.config.radio_listen_heating_rate_cps * dt_s
        elif comms_mode == "BEACON":
            delta += self.config.radio_beacon_heating_rate_cps * dt_s
        elif comms_mode == "LOW_RATE_TX":
            delta += self.config.radio_low_rate_tx_heating_rate_cps * dt_s
        elif comms_mode == "HIGH_RATE_TX":
            delta += self.config.radio_high_rate_tx_heating_rate_cps * dt_s

        # Transient action heat contribution
        if action_profile is not None and action_profile.radio_use:
            delta += 0.25 * action_profile.heat_delta_c

        return delta

    # -----------------------------------------------------
    # Thermal policy / heater logic
    # -----------------------------------------------------

    def _update_thermal_mode(self, state: SpacecraftState) -> None:
        """
        Updates thermal mode and simple heater hysteresis behavior.
        """
        bt = state.thermal.battery_temp_c
        pt = state.thermal.payload_temp_c
        bus = state.thermal.bus_temp_c

        # Heater hysteresis
        if bt <= self.config.battery_heat_enable_below_c:
            state.thermal.battery_heater_on = True
        elif bt >= self.config.battery_heat_disable_above_c:
            state.thermal.battery_heater_on = False

        if pt <= self.config.payload_heat_enable_below_c:
            state.thermal.payload_heater_on = True
        elif pt >= self.config.payload_heat_disable_above_c:
            state.thermal.payload_heater_on = False

        # Mode selection
        hottest = max(bt, pt, bus)
        coldest = min(bt, pt, bus)

        if hottest >= self.config.hot_soak_threshold_c:
            state.thermal.mode = ThermalMode.HOT_SOAK
            return

        if coldest <= self.config.cold_soak_threshold_c:
            if state.thermal.battery_heater_on:
                state.thermal.mode = ThermalMode.BATTERY_HEATING
            elif state.thermal.payload_heater_on:
                state.thermal.mode = ThermalMode.PAYLOAD_HEATING
            else:
                state.thermal.mode = ThermalMode.COLD_SOAK
            return

        if state.thermal.thermal_violation:
            state.thermal.mode = ThermalMode.THERMAL_PROTECT
            return

        if state.mode == state.mode.SURVIVAL:
            state.thermal.mode = ThermalMode.SURVIVAL
            return

        state.thermal.mode = ThermalMode.NOMINAL


# =========================================================
# Functional helpers
# =========================================================

def initialize_thermal_state(
    state: SpacecraftState,
    config: ThermalConfig = DEFAULT_THERMAL_CONFIG,
) -> None:
    subsystem = ThermalSubsystem(config=config)
    subsystem.initialize(state)


def update_thermal_state(
    state: SpacecraftState,
    dt_s: float,
    action: Optional[Action] = None,
    action_profile: Optional[ActionEffectProfile] = None,
    config: ThermalConfig = DEFAULT_THERMAL_CONFIG,
) -> ThermalBreakdown:
    subsystem = ThermalSubsystem(config=config)
    return subsystem.step(
        state=state,
        dt_s=dt_s,
        action=action,
        action_profile=action_profile,
    )


# =========================================================
# Smoke test helper
# =========================================================

def thermal_smoke_summary(
    state: SpacecraftState,
    dt_s: float = 5.0,
    steps: int = 30,
    config: ThermalConfig = DEFAULT_THERMAL_CONFIG,
) -> Dict[str, float]:
    """
    Quick thermal sanity-check helper.
    """
    subsystem = ThermalSubsystem(config=config)

    min_bus = state.thermal.bus_temp_c
    max_bus = state.thermal.bus_temp_c
    min_batt = state.thermal.battery_temp_c
    max_batt = state.thermal.battery_temp_c
    min_payload = state.thermal.payload_temp_c
    max_payload = state.thermal.payload_temp_c
    min_radio = state.thermal.radio_temp_c
    max_radio = state.thermal.radio_temp_c

    for _ in range(steps):
        subsystem.step(state=state, dt_s=dt_s)

        min_bus = min(min_bus, state.thermal.bus_temp_c)
        max_bus = max(max_bus, state.thermal.bus_temp_c)

        min_batt = min(min_batt, state.thermal.battery_temp_c)
        max_batt = max(max_batt, state.thermal.battery_temp_c)

        min_payload = min(min_payload, state.thermal.payload_temp_c)
        max_payload = max(max_payload, state.thermal.payload_temp_c)

        min_radio = min(min_radio, state.thermal.radio_temp_c)
        max_radio = max(max_radio, state.thermal.radio_temp_c)

    return {
        "final_bus_temp_c": state.thermal.bus_temp_c,
        "final_battery_temp_c": state.thermal.battery_temp_c,
        "final_payload_temp_c": state.thermal.payload_temp_c,
        "final_radio_temp_c": state.thermal.radio_temp_c,
        "min_bus_temp_c": min_bus,
        "max_bus_temp_c": max_bus,
        "min_battery_temp_c": min_batt,
        "max_battery_temp_c": max_batt,
        "min_payload_temp_c": min_payload,
        "max_payload_temp_c": max_payload,
        "min_radio_temp_c": min_radio,
        "max_radio_temp_c": max_radio,
        "battery_heater_on": float(state.thermal.battery_heater_on),
        "payload_heater_on": float(state.thermal.payload_heater_on),
    }