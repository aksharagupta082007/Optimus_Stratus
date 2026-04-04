from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional

from models.action_space import ActionEffectProfile
from models.enums import Action, CommsMode, GroundPassState, LinkQuality
from models.state_models import SpacecraftState


# =========================================================
# Configuration
# =========================================================

@dataclass(frozen=True)
class CommsConfig:
    """
    Operational communications subsystem model for a LEO CubeSat.

    This is a mission-operations comms simulator, not a full RF link-budget solver.
    It is designed to model:

    - pass visibility gating
    - pass state and elevation-based link quality
    - mode transitions for listen / prep / beacon / downlink
    - low-rate and high-rate downlink throughput
    - packet success probability
    - queue depletion and byte counters

    Units:
    - data rate: Mbps
    - data volume: bytes / MB
    - time: s
    - elevation: deg
    - range: km
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


DEFAULT_COMMS_CONFIG = CommsConfig()


# =========================================================
# Helpers
# =========================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# =========================================================
# Breakdown
# =========================================================

@dataclass(frozen=True)
class CommsBreakdown:
    """
    Detailed comms update report for debugging, testing, and reward shaping.
    """
    gs_visible: bool
    pass_state: GroundPassState
    link_quality: LinkQuality
    gs_elevation_deg: float
    range_km: float
    current_rate_mbps: float
    packet_success_prob: float
    bytes_downlinked_this_step: int
    total_bytes_downlinked: int
    queue_before_mb: float
    queue_after_mb: float
    pass_time_remaining_s: float


# =========================================================
# Communications subsystem
# =========================================================

class CommsSubsystem:
    """
    Operational communications subsystem.

    Responsibilities:
    - derive pass visibility and link quality from orbit state
    - compute achievable throughput from mode + elevation + attitude quality
    - update downlink queue and byte counters
    - maintain comms state for the environment
    """

    def __init__(self, config: CommsConfig = DEFAULT_COMMS_CONFIG):
        self.config = config
        self.rng = random.Random(config.random_seed)

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def initialize(self, state: SpacecraftState) -> None:
        state.comms.low_rate_mbps = self.config.low_rate_mbps
        state.comms.high_rate_mbps = self.config.high_rate_mbps
        state.comms.current_rate_mbps = 0.0
        state.comms.bytes_downlinked_this_step = 0
        state.comms.total_bytes_downlinked = 0
        self._refresh_pass_geometry(state)

    def step(
        self,
        state: SpacecraftState,
        dt_s: float,
        action: Optional[Action] = None,
        action_profile: Optional[ActionEffectProfile] = None,
    ) -> CommsBreakdown:
        """
        Advance comms state by one step.

        The environment should already have advanced orbit state before calling this.
        """
        queue_before_mb = state.cdh.downlink_queue_mb

        # Keep geometry synced from orbit
        self._refresh_pass_geometry(state)

        # Update comms mode from explicit action if needed
        self._apply_action_mode_hint(state, action)

        # Tx/Rx flags from current mode
        self._update_radio_flags(state)

        # Compute usable throughput
        current_rate_mbps = self._compute_effective_rate_mbps(state)
        current_rate_mbps = clamp(
            current_rate_mbps,
            self.config.min_rate_mbps,
            self.config.max_rate_mbps,
        )
        state.comms.current_rate_mbps = current_rate_mbps

        packet_success_prob = self._compute_packet_success_probability(state)
        state.comms.packet_success_prob = packet_success_prob

        bytes_downlinked = 0

        # Only actual TX modes consume queue
        if state.comms.tx_enabled and state.comms.gs_visible and current_rate_mbps > 0.0:
            outage = self._sample_outage(state.comms.link_quality)

            if not outage:
                ideal_bytes = int((current_rate_mbps * 1_000_000 / 8.0) * dt_s)
                successful_bytes = int(ideal_bytes * packet_success_prob)

                queue_bytes = self._mb_to_bytes(state.cdh.downlink_queue_mb)
                bytes_downlinked = min(successful_bytes, queue_bytes)

                state.cdh.downlink_queue_mb = self._bytes_to_mb(max(0, queue_bytes - bytes_downlinked))
                state.cdh.total_useful_mb_downlinked += self._bytes_to_mb(bytes_downlinked)

        state.comms.bytes_downlinked_this_step = bytes_downlinked
        state.comms.total_bytes_downlinked += bytes_downlinked

        queue_after_mb = state.cdh.downlink_queue_mb

        return CommsBreakdown(
            gs_visible=state.comms.gs_visible,
            pass_state=state.comms.pass_state,
            link_quality=state.comms.link_quality,
            gs_elevation_deg=state.comms.gs_elevation_deg,
            range_km=state.comms.range_km,
            current_rate_mbps=state.comms.current_rate_mbps,
            packet_success_prob=state.comms.packet_success_prob,
            bytes_downlinked_this_step=state.comms.bytes_downlinked_this_step,
            total_bytes_downlinked=state.comms.total_bytes_downlinked,
            queue_before_mb=queue_before_mb,
            queue_after_mb=queue_after_mb,
            pass_time_remaining_s=state.comms.pass_time_remaining_s,
        )

    # -----------------------------------------------------
    # Pass geometry refresh
    # -----------------------------------------------------

    def _refresh_pass_geometry(self, state: SpacecraftState) -> None:
        """
        Derive pass visibility directly from orbit phase and pass windows.

        This duplicates a small amount of logic from the orbit subsystem so the
        comms module remains self-contained, but it uses the already-updated orbit
        state as the source of truth.
        """
        phase = state.orbit.orbit_phase

        # Default no-pass state
        visible = False
        elevation = 0.0
        pass_time_remaining_s = 0.0

        # Use orbit pass windows if available via internal config hints.
        # Since orbit.py owns the actual pass structure, we infer visibility from
        # orbit_state land/phase-compatible patterns using a smooth proxy.
        #
        # For robustness in this subsystem, we use an elevation dome centered on two
        # generic passes per orbit.
        pass_centers = (0.18, 0.56)
        pass_half_widths = (0.045, 0.055)

        best_progress = None
        best_center = None
        best_half_width = None

        for center, half_width in zip(pass_centers, pass_half_widths):
            dist = self._circular_phase_distance(phase, center)
            if dist <= half_width:
                progress = dist / max(half_width, 1e-9)
                if best_progress is None or progress < best_progress:
                    best_progress = progress
                    best_center = center
                    best_half_width = half_width

        if best_progress is not None:
            visible = True
            elevation = 85.0 * math.sqrt(max(0.0, 1.0 - best_progress**2))
            exit_phase = (best_center + best_half_width) % 1.0
            delta_phase = (exit_phase - phase) % 1.0
            pass_time_remaining_s = delta_phase * state.orbit.orbit_period_s

        pass_state = self._classify_pass_state(elevation if visible else 0.0)
        link_quality = self._classify_link_quality(elevation if visible else 0.0)
        range_km = self._estimate_slant_range_km(elevation if visible else 0.0)

        state.comms.gs_visible = visible
        state.comms.pass_state = pass_state
        state.comms.link_quality = link_quality
        state.comms.gs_elevation_deg = elevation
        state.comms.range_km = range_km
        state.comms.pass_time_remaining_s = pass_time_remaining_s

    # -----------------------------------------------------
    # Mode control
    # -----------------------------------------------------

    def _apply_action_mode_hint(self, state: SpacecraftState, action: Optional[Action]) -> None:
        if action is None:
            return

        mapping = {
            Action.PREPARE_DOWNLINK: CommsMode.PASS_PREP,
            Action.DOWNLINK_LOW_RATE: CommsMode.LOW_RATE_TX,
            Action.DOWNLINK_HIGH_RATE: CommsMode.HIGH_RATE_TX,
            Action.SEND_BEACON: CommsMode.BEACON,
            Action.RESET_COMMS: CommsMode.OFF,
            Action.ENTER_SAFE_MODE: CommsMode.LISTEN,
        }
        new_mode = mapping.get(action)
        if new_mode is not None:
            state.comms.mode = new_mode

    def _update_radio_flags(self, state: SpacecraftState) -> None:
        mode = state.comms.mode

        if mode == CommsMode.OFF:
            state.comms.tx_enabled = False
            state.comms.rx_enabled = False
        elif mode in (CommsMode.LISTEN, CommsMode.RX_ONLY, CommsMode.PASS_PREP):
            state.comms.tx_enabled = False
            state.comms.rx_enabled = True
        elif mode == CommsMode.BEACON:
            state.comms.tx_enabled = True
            state.comms.rx_enabled = False
        elif mode in (CommsMode.LOW_RATE_TX, CommsMode.HIGH_RATE_TX):
            state.comms.tx_enabled = True
            state.comms.rx_enabled = False
        elif mode == CommsMode.TXRX:
            state.comms.tx_enabled = True
            state.comms.rx_enabled = True
        else:
            state.comms.tx_enabled = False
            state.comms.rx_enabled = False

    # -----------------------------------------------------
    # Throughput model
    # -----------------------------------------------------

    def _compute_effective_rate_mbps(self, state: SpacecraftState) -> float:
        """
        Compute achievable throughput from:
        - radio mode
        - visibility
        - elevation / link quality
        - ADCS posture and settling
        - pointing error
        """
        mode = state.comms.mode

        if not state.comms.gs_visible:
            if mode == CommsMode.BEACON:
                return self.config.beacon_rate_mbps
            return 0.0

        base_rate = 0.0
        if mode == CommsMode.LOW_RATE_TX:
            base_rate = self.config.low_rate_mbps
        elif mode == CommsMode.HIGH_RATE_TX:
            base_rate = self.config.high_rate_mbps
        elif mode == CommsMode.BEACON:
            base_rate = self.config.beacon_rate_mbps
        elif mode in (CommsMode.LISTEN, CommsMode.RX_ONLY, CommsMode.TXRX, CommsMode.PASS_PREP):
            base_rate = self.config.rx_housekeeping_rate_mbps
        else:
            return 0.0

        link_factor = self._link_quality_rate_factor(state.comms.link_quality)
        attitude_factor = self._attitude_rate_factor(state)
        pointing_factor = self._pointing_rate_factor(state, high_rate=(mode == CommsMode.HIGH_RATE_TX))

        effective = base_rate * link_factor * attitude_factor * pointing_factor

        # High-rate TX may be disallowed under bad ADCS conditions
        if mode == CommsMode.HIGH_RATE_TX:
            if self.config.require_ground_tracking_for_high_rate and state.adcs.mode.name != "GROUND_TRACKING":
                effective *= 0.0
            if self.config.require_good_pointing_for_high_rate and (
                state.adcs.pointing_error_deg > self.config.high_rate_allowed_pointing_error_deg
            ):
                effective *= 0.0

        return effective

    def _link_quality_rate_factor(self, quality: LinkQuality) -> float:
        if quality == LinkQuality.POOR:
            return self.config.poor_quality_rate_factor
        if quality == LinkQuality.MARGINAL:
            return self.config.marginal_quality_rate_factor
        if quality == LinkQuality.GOOD:
            return self.config.good_quality_rate_factor
        if quality == LinkQuality.EXCELLENT:
            return self.config.excellent_quality_rate_factor
        return 0.0

    def _attitude_rate_factor(self, state: SpacecraftState) -> float:
        if state.adcs.mode.name == "GROUND_TRACKING":
            factor = self.config.ground_tracking_bonus_factor
        elif state.adcs.mode.name == "INERTIAL_HOLD":
            factor = self.config.inertial_hold_factor
        else:
            factor = self.config.wrong_attitude_factor

        if self.config.require_pointing_settle_for_downlink and state.adcs.slew_time_remaining_s > 0.0:
            factor *= self.config.unsettled_factor

        return factor

    def _pointing_rate_factor(self, state: SpacecraftState, high_rate: bool) -> float:
        err = state.adcs.pointing_error_deg
        threshold = (
            self.config.high_rate_allowed_pointing_error_deg
            if high_rate
            else self.config.low_rate_allowed_pointing_error_deg
        )

        if err <= 0.2:
            return 1.0
        if err >= threshold:
            return 0.0

        # Linear taper between ideal and allowed threshold
        return clamp(1.0 - (err - 0.2) / max(threshold - 0.2, 1e-9), 0.0, 1.0)

    # -----------------------------------------------------
    # Reliability model
    # -----------------------------------------------------

    def _compute_packet_success_probability(self, state: SpacecraftState) -> float:
        q = state.comms.link_quality

        if q == LinkQuality.POOR:
            p = self.config.poor_quality_success_prob
        elif q == LinkQuality.MARGINAL:
            p = self.config.marginal_quality_success_prob
        elif q == LinkQuality.GOOD:
            p = self.config.good_quality_success_prob
        elif q == LinkQuality.EXCELLENT:
            p = self.config.excellent_quality_success_prob
        else:
            p = 0.0

        # ADCS penalties
        if state.adcs.slew_time_remaining_s > 0.0:
            p *= 0.75

        if state.adcs.pointing_error_deg > 2.0:
            p *= 0.80
        elif state.adcs.pointing_error_deg > 1.0:
            p *= 0.90

        if state.adcs.wheels_saturated:
            p *= 0.92

        return clamp(p, 0.0, 1.0)

    def _sample_outage(self, quality: LinkQuality) -> bool:
        if quality == LinkQuality.POOR:
            p = self.config.outage_probability_poor_link
        elif quality == LinkQuality.MARGINAL:
            p = self.config.outage_probability_marginal_link
        elif quality == LinkQuality.GOOD:
            p = self.config.outage_probability_good_link
        elif quality == LinkQuality.EXCELLENT:
            p = self.config.outage_probability_excellent_link
        else:
            p = 1.0

        return self.rng.random() < p

    # -----------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------

    def _classify_pass_state(self, elevation_deg: float) -> GroundPassState:
        if elevation_deg <= 0.0:
            return GroundPassState.NONE
        if elevation_deg < 8.0:
            return GroundPassState.ACQUIRE
        if elevation_deg < 25.0:
            return GroundPassState.LOW_ELEVATION
        if elevation_deg < 55.0:
            return GroundPassState.MID_ELEVATION
        return GroundPassState.HIGH_ELEVATION

    def _classify_link_quality(self, elevation_deg: float) -> LinkQuality:
        if elevation_deg <= 0.0:
            return LinkQuality.NONE
        if elevation_deg < self.config.low_elev_deg:
            return LinkQuality.POOR
        if elevation_deg < self.config.mid_elev_deg:
            return LinkQuality.MARGINAL
        if elevation_deg < self.config.high_elev_deg:
            return LinkQuality.GOOD
        return LinkQuality.EXCELLENT

    def _estimate_slant_range_km(self, elevation_deg: float) -> float:
        """
        Lightweight range proxy:
        - high elevation => shorter range
        - low elevation => longer range
        """
        if elevation_deg <= 0.0:
            return self.config.max_slant_range_km

        t = clamp(elevation_deg / 90.0, 0.0, 1.0)
        # More curved than linear, to reflect stronger range reduction near zenith
        shaped = t**0.65
        return self.config.max_slant_range_km - shaped * (
            self.config.max_slant_range_km - self.config.min_slant_range_km
        )

    def _circular_phase_distance(self, a: float, b: float) -> float:
        d = abs((a % 1.0) - (b % 1.0))
        return min(d, 1.0 - d)

    # -----------------------------------------------------
    # Unit conversion helpers
    # -----------------------------------------------------

    def _mb_to_bytes(self, mb: float) -> int:
        return int(max(0.0, mb) * self.config.bytes_per_mb_decimal)

    def _bytes_to_mb(self, byte_count: int) -> float:
        return max(0.0, byte_count) / self.config.bytes_per_mb_decimal


# =========================================================
# Functional helpers
# =========================================================

def initialize_comms_state(
    state: SpacecraftState,
    config: CommsConfig = DEFAULT_COMMS_CONFIG,
) -> None:
    subsystem = CommsSubsystem(config=config)
    subsystem.initialize(state)


def update_comms_state(
    state: SpacecraftState,
    dt_s: float,
    action: Optional[Action] = None,
    action_profile: Optional[ActionEffectProfile] = None,
    config: CommsConfig = DEFAULT_COMMS_CONFIG,
) -> CommsBreakdown:
    subsystem = CommsSubsystem(config=config)
    return subsystem.step(
        state=state,
        dt_s=dt_s,
        action=action,
        action_profile=action_profile,
    )


# =========================================================
# Smoke test helper
# =========================================================

def comms_smoke_summary(
    state: SpacecraftState,
    dt_s: float = 5.0,
    steps: int = 20,
    action: Optional[Action] = None,
    config: CommsConfig = DEFAULT_COMMS_CONFIG,
) -> Dict[str, float]:
    """
    Quick communications sanity-check helper.
    """
    subsystem = CommsSubsystem(config=config)

    visible_steps = 0
    max_rate = 0.0
    max_elev = 0.0
    total_bytes = state.comms.total_bytes_downlinked
    queue_start = state.cdh.downlink_queue_mb

    for i in range(steps):
        breakdown = subsystem.step(
            state=state,
            dt_s=dt_s,
            action=action if i == 0 else None,
        )
        if state.comms.gs_visible:
            visible_steps += 1
        max_rate = max(max_rate, breakdown.current_rate_mbps)
        max_elev = max(max_elev, breakdown.gs_elevation_deg)

    return {
        "visible_fraction": visible_steps / max(steps, 1),
        "max_rate_mbps": max_rate,
        "max_elevation_deg": max_elev,
        "queue_start_mb": queue_start,
        "queue_end_mb": state.cdh.downlink_queue_mb,
        "bytes_downlinked_total": state.comms.total_bytes_downlinked - total_bytes,
        "current_link_quality": float(int(state.comms.link_quality)),
        "current_pass_state": float(int(state.comms.pass_state)),
    }