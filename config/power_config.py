from dataclasses import dataclass


@dataclass(frozen=True)
class PowerConfig:
    """
    EPS / power subsystem configuration for a realistic small LEO CubeSat.
    Matches the expectations of subsystems/power.py.
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

    sun_pointing_bonus_factor: float = 1.08
    nadir_pointing_factor: float = 0.88
    ground_tracking_factor: float = 0.82
    inertial_hold_factor: float = 0.90
    safe_posture_factor: float = 0.92

    beta_angle_gain: float = 0.10

    # -----------------------------------------------------
    # Base avionics / housekeeping load
    # -----------------------------------------------------
    base_housekeeping_load_w: float = 2.2
    safe_mode_load_w: float = 1.6
    survival_mode_load_w: float = 1.3
    eclipse_power_save_load_w: float = 1.8

    adcs_background_load_w: float = 0.7
    payload_background_load_w: float = 0.2
    comms_background_load_w: float = 0.15
    cdh_background_load_w: float = 0.35
    thermal_background_load_w: float = 0.1

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
    # Battery efficiency / charge-discharge behavior
    # -----------------------------------------------------
    charge_efficiency: float = 0.93
    discharge_efficiency: float = 0.95

    taper_start_soc_pct: float = 85.0
    taper_end_soc_pct: float = 100.0
    taper_min_factor: float = 0.15

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
