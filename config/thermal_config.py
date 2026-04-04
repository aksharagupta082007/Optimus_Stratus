from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalConfig:
    """
    Operational thermal model configuration for a small Earth-observation CubeSat.
    Matches the expectations of subsystems/thermal.py.
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
    # -----------------------------------------------------
    bus_time_constant_s: float = 900.0
    battery_time_constant_s: float = 1400.0
    payload_time_constant_s: float = 1000.0
    radio_time_constant_s: float = 700.0

    # -----------------------------------------------------
    # Internal heating rates (deg C / sec at full activity)
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
    # Heater effects
    # -----------------------------------------------------
    battery_heater_rate_cps: float = 0.0120
    payload_heater_rate_cps: float = 0.0100

    # -----------------------------------------------------
    # Cross-coupling factors
    # -----------------------------------------------------
    bus_to_battery_coupling: float = 0.08
    bus_to_payload_coupling: float = 0.10
    bus_to_radio_coupling: float = 0.07

    payload_to_bus_coupling: float = 0.05
    radio_to_bus_coupling: float = 0.04
    battery_to_bus_coupling: float = 0.03

    # -----------------------------------------------------
    # Net-power influence
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
