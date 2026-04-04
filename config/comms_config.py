from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CommsConfig:
    """
    Operational communications subsystem model for a LEO CubeSat.
    Matches the expectations of subsystems/comms.py.
    """

    # -----------------------------------------------------
    # Nominal rates
    # -----------------------------------------------------
    low_rate_mbps: float = 0.25
    high_rate_mbps: float = 2.00
    beacon_rate_mbps: float = 0.02
    rx_housekeeping_rate_mbps: float = 0.01

    # -----------------------------------------------------
    # Pass/elevation quality thresholds
    # -----------------------------------------------------
    acquire_min_elev_deg: float = 3.0
    low_elev_deg: float = 10.0
    mid_elev_deg: float = 30.0
    high_elev_deg: float = 60.0

    # -----------------------------------------------------
    # Effective throughput multipliers by link quality
    # -----------------------------------------------------
    poor_quality_rate_factor: float = 0.30
    marginal_quality_rate_factor: float = 0.55
    good_quality_rate_factor: float = 0.82
    excellent_quality_rate_factor: float = 0.95

    # -----------------------------------------------------
    # Packet success probability by link quality
    # -----------------------------------------------------
    poor_quality_success_prob: float = 0.55
    marginal_quality_success_prob: float = 0.78
    good_quality_success_prob: float = 0.92
    excellent_quality_success_prob: float = 0.98

    # -----------------------------------------------------
    # Geometry / range proxy
    # -----------------------------------------------------
    min_slant_range_km: float = 550.0
    max_slant_range_km: float = 2200.0

    # -----------------------------------------------------
    # ADCS coupling
    # -----------------------------------------------------
    require_ground_tracking_for_high_rate: bool = True
    require_good_pointing_for_high_rate: bool = True
    require_pointing_settle_for_downlink: bool = True

    high_rate_allowed_pointing_error_deg: float = 1.2
    low_rate_allowed_pointing_error_deg: float = 4.0

    ground_tracking_bonus_factor: float = 1.00
    inertial_hold_factor: float = 0.80
    wrong_attitude_factor: float = 0.45
    unsettled_factor: float = 0.60

    # -----------------------------------------------------
    # Reliability / stochasticity
    # -----------------------------------------------------
    random_seed: Optional[int] = None
    outage_probability_poor_link: float = 0.06
    outage_probability_marginal_link: float = 0.02
    outage_probability_good_link: float = 0.005
    outage_probability_excellent_link: float = 0.001

    # -----------------------------------------------------
    # Byte conversion
    # -----------------------------------------------------
    bytes_per_mb_decimal: int = 1_000_000

    # -----------------------------------------------------
    # Numerical guards
    # -----------------------------------------------------
    min_rate_mbps: float = 0.0
    max_rate_mbps: float = 5.0
