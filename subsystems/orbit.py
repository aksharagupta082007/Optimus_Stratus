from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from models.enums import GroundPassState, LinkQuality, SunlightState, TargetOpportunity
from models.state_models import OrbitState


# =========================================================
# Configuration
# =========================================================

@dataclass(frozen=True)
class OrbitConfig:
    """
    Operational orbit simulator configuration for a sun-synchronous
    Earth-observation CubeSat in LEO.

    This is not a full astrodynamics propagator. It is a mission-grade
    operational orbit model intended for RL and planning:
    - orbit phase progression
    - sunlit/eclipse segmentation
    - target opportunity windows
    - ground-station pass windows
    - simple latitude/longitude evolution
    - sun elevation proxy
    """

    # -----------------------------------------------------
    # Core orbit
    # -----------------------------------------------------
    orbit_period_s: float = 5400.0
    altitude_km: float = 525.0
    inclination_deg: float = 97.5

    # -----------------------------------------------------
    # Lighting geometry
    # -----------------------------------------------------
    sunlit_fraction: float = 0.62
    eclipse_center_phase: float = 0.78

    # Daylight imaging proxy
    min_sun_elevation_for_imaging_deg: float = 10.0
    peak_sun_elevation_deg: float = 65.0
    eclipse_min_sun_elevation_deg: float = -20.0

    # -----------------------------------------------------
    # Ground pass windows (phase centers and widths)
    # -----------------------------------------------------
    pass_phase_centers: Tuple[float, ...] = (0.18, 0.56)
    pass_half_widths: Tuple[float, ...] = (0.045, 0.055)

    # -----------------------------------------------------
    # Target opportunity windows
    # -----------------------------------------------------
    target_phase_centers: Tuple[float, ...] = (0.10, 0.33, 0.67, 0.90)
    target_half_widths: Tuple[float, ...] = (0.04, 0.05, 0.05, 0.035)

    # -----------------------------------------------------
    # Stochastic opportunity generation
    # -----------------------------------------------------
    base_target_visibility_prob: float = 0.85
    high_value_target_prob: float = 0.18
    poor_light_target_prob: float = 0.10

    # -----------------------------------------------------
    # Simple ground-track proxy
    # -----------------------------------------------------
    latitude_amplitude_deg: float = 82.0
    longitude_rate_deg_per_orbit: float = 22.5

    # -----------------------------------------------------
    # Randomness / reproducibility
    # -----------------------------------------------------
    random_seed: Optional[int] = None


DEFAULT_ORBIT_CONFIG = OrbitConfig()


# =========================================================
# Utility helpers
# =========================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_phase(x: float) -> float:
    return x % 1.0


def wrap_longitude_deg(lon: float) -> float:
    """
    Wrap longitude to [-180, 180).
    """
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    return wrapped


def circular_phase_distance(a: float, b: float) -> float:
    """
    Distance on a unit circle between two normalized phases [0,1).
    """
    d = abs(wrap_phase(a) - wrap_phase(b))
    return min(d, 1.0 - d)


# =========================================================
# Derived pass / target info containers
# =========================================================

@dataclass(frozen=True)
class GroundPassInfo:
    visible: bool
    pass_state: GroundPassState
    elevation_deg: float
    link_quality: LinkQuality
    time_to_next_pass_s: float
    time_remaining_in_pass_s: float


@dataclass(frozen=True)
class TargetInfo:
    over_target: bool
    target_opportunity: TargetOpportunity
    quality_score: float


# =========================================================
# Orbit propagator / operational model
# =========================================================

class OrbitPropagator:
    """
    Operational LEO orbit model for mission simulation.

    Responsibilities:
    - advance orbit phase and orbit index
    - update sunlight/eclipse
    - compute sun elevation proxy
    - update simple latitude/longitude ground track
    - compute ground station pass visibility/elevation/link quality
    - compute target opportunity windows

    Notes:
    - This is deliberately mission-operational, not high-fidelity orbital mechanics.
    - It is designed for RL environment realism and stable simulation.
    """

    def __init__(self, config: OrbitConfig = DEFAULT_ORBIT_CONFIG):
        self.config = config
        self.rng = random.Random(config.random_seed)

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def reset(
        self,
        orbit_state: OrbitState,
        initial_phase: float = 0.0,
        orbit_index: int = 0,
    ) -> OrbitState:
        orbit_state.orbit_period_s = self.config.orbit_period_s
        orbit_state.altitude_km = self.config.altitude_km
        orbit_state.inclination_deg = self.config.inclination_deg
        orbit_state.orbit_phase = wrap_phase(initial_phase)

        self._update_orbit_index_safe(orbit_state, orbit_index)
        self._update_geometry(orbit_state)
        return orbit_state

    def step(
        self,
        orbit_state: OrbitState,
        dt_s: float,
        time_orbit_index: Optional[int] = None,
    ) -> OrbitState:
        """
        Advance the operational orbit state by dt_s seconds.
        """
        old_phase = orbit_state.orbit_phase
        delta_phase = dt_s / max(self.config.orbit_period_s, 1e-9)
        new_phase = wrap_phase(old_phase + delta_phase)

        wrapped = new_phase < old_phase
        orbit_state.orbit_phase = new_phase

        if time_orbit_index is not None:
            self._update_orbit_index_safe(orbit_state, time_orbit_index)
        elif wrapped:
            # Best-effort increment if no external time index is supplied.
            # OrbitState does not store orbit index, so we keep geometry coherent
            # and let env/time layer own the authoritative counter.
            pass

        self._update_geometry(orbit_state)
        return orbit_state

    def get_ground_pass_info(self, orbit_state: OrbitState) -> GroundPassInfo:
        return self._compute_ground_pass_info(orbit_state.orbit_phase)

    def get_target_info(self, orbit_state: OrbitState) -> TargetInfo:
        return self._compute_target_info(
            orbit_phase=orbit_state.orbit_phase,
            sunlit_state=orbit_state.sunlight_state,
            sun_elevation_deg=orbit_state.sun_elevation_deg,
        )

    def time_to_next_pass_s(self, orbit_phase: float) -> float:
        return self._time_to_next_window_s(
            orbit_phase=orbit_phase,
            centers=self.config.pass_phase_centers,
            half_widths=self.config.pass_half_widths,
            return_zero_if_inside=True,
        )

    def time_to_next_target_s(self, orbit_phase: float) -> float:
        return self._time_to_next_window_s(
            orbit_phase=orbit_phase,
            centers=self.config.target_phase_centers,
            half_widths=self.config.target_half_widths,
            return_zero_if_inside=True,
        )

    # -----------------------------------------------------
    # Internal geometry update
    # -----------------------------------------------------

    def _update_geometry(self, orbit_state: OrbitState) -> None:
        phase = orbit_state.orbit_phase

        orbit_state.sunlight_state = self._compute_sunlight_state(phase)
        orbit_state.sun_elevation_deg = self._compute_sun_elevation_deg(
            phase,
            orbit_state.sunlight_state,
        )

        lat_deg, lon_deg = self._compute_ground_track(
            phase=phase,
            orbit_index=getattr(orbit_state, "_orbit_index_internal", 0),
        )
        orbit_state.latitude_deg = lat_deg
        orbit_state.longitude_deg = lon_deg

        orbit_state.beta_angle_deg = self._compute_beta_angle_deg(phase)

        target_info = self._compute_target_info(
            orbit_phase=phase,
            sunlit_state=orbit_state.sunlight_state,
            sun_elevation_deg=orbit_state.sun_elevation_deg,
        )
        orbit_state.over_target = target_info.over_target
        orbit_state.target_opportunity = target_info.target_opportunity
        orbit_state.land_fraction_hint = target_info.quality_score

    def _update_orbit_index_safe(self, orbit_state: OrbitState, orbit_index: int) -> None:
        """
        OrbitState does not currently expose orbit_index directly, so we attach
        an internal attribute for geometry continuity. This keeps this subsystem
        decoupled from SimTime while still allowing realistic longitude drift.
        """
        setattr(orbit_state, "_orbit_index_internal", int(orbit_index))

    # -----------------------------------------------------
    # Lighting model
    # -----------------------------------------------------

    def _compute_sunlight_state(self, phase: float) -> SunlightState:
        """
        Defines a single eclipse arc centered near eclipse_center_phase.
        """
        eclipse_width = 1.0 - self.config.sunlit_fraction
        half_eclipse = eclipse_width / 2.0
        dist = circular_phase_distance(phase, self.config.eclipse_center_phase)

        if dist <= half_eclipse:
            return SunlightState.ECLIPSE
        return SunlightState.SUNLIT

    def _compute_sun_elevation_deg(
        self,
        phase: float,
        sunlight_state: SunlightState,
    ) -> float:
        """
        Proxy solar elevation over the observed ground track.
        Designed to be:
        - positive and varying in sunlit arc
        - negative in eclipse
        - smooth enough for RL
        """
        # phase mapped to cosine-like daylight shape
        daylight_wave = 0.5 * (1.0 + math.cos(2.0 * math.pi * (phase - 0.25)))

        if sunlight_state == SunlightState.SUNLIT:
            sun_elev = (
                self.config.min_sun_elevation_for_imaging_deg
                + daylight_wave
                * (self.config.peak_sun_elevation_deg - self.config.min_sun_elevation_for_imaging_deg)
            )
            return clamp(sun_elev, self.config.min_sun_elevation_for_imaging_deg, self.config.peak_sun_elevation_deg)

        eclipse_wave = 0.5 * (1.0 + math.cos(2.0 * math.pi * (phase - self.config.eclipse_center_phase)))
        sun_elev = self.config.eclipse_min_sun_elevation_deg * (0.5 + 0.5 * eclipse_wave)
        return clamp(sun_elev, self.config.eclipse_min_sun_elevation_deg, 0.0)

    def _compute_beta_angle_deg(self, phase: float) -> float:
        """
        Simplified beta-angle-like proxy. Keeps thermal/power geometry varying gently.
        """
        return 15.0 + 10.0 * math.sin(2.0 * math.pi * phase)

    # -----------------------------------------------------
    # Ground track model
    # -----------------------------------------------------

    def _compute_ground_track(self, phase: float, orbit_index: int) -> Tuple[float, float]:
        """
        Very lightweight ground track proxy.
        - latitude oscillates sinusoidally
        - longitude drifts per orbit
        """
        lat = self.config.latitude_amplitude_deg * math.sin(2.0 * math.pi * phase)

        # The 0.25 shift just makes initial longitude less trivial.
        lon = (
            -160.0
            + 360.0 * wrap_phase(phase + 0.25)
            + orbit_index * self.config.longitude_rate_deg_per_orbit
        )
        lon = wrap_longitude_deg(lon)

        return lat, lon

    # -----------------------------------------------------
    # Ground pass model
    # -----------------------------------------------------

    def _compute_ground_pass_info(self, orbit_phase: float) -> GroundPassInfo:
        """
        A pass exists when orbit phase falls inside one of the configured phase windows.
        Elevation is modeled as a smooth dome over the pass window.
        """
        best_pass_idx = None
        best_local_progress = None
        best_local_half_width = None
        best_distance = None

        for idx, (center, half_width) in enumerate(
            zip(self.config.pass_phase_centers, self.config.pass_half_widths)
        ):
            dist = circular_phase_distance(orbit_phase, center)
            if dist <= half_width:
                progress = dist / max(half_width, 1e-9)
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_pass_idx = idx
                    best_local_progress = progress
                    best_local_half_width = half_width

        if best_pass_idx is None:
            return GroundPassInfo(
                visible=False,
                pass_state=GroundPassState.NONE,
                elevation_deg=0.0,
                link_quality=LinkQuality.NONE,
                time_to_next_pass_s=self._time_to_next_window_s(
                    orbit_phase=orbit_phase,
                    centers=self.config.pass_phase_centers,
                    half_widths=self.config.pass_half_widths,
                    return_zero_if_inside=False,
                ),
                time_remaining_in_pass_s=0.0,
            )

        # Smooth elevation dome: 0 at edges, ~85 deg at center
        elevation_deg = 85.0 * math.sqrt(max(0.0, 1.0 - best_local_progress**2))
        pass_state = self._classify_pass_state(elevation_deg)
        link_quality = self._classify_link_quality(elevation_deg)

        center = self.config.pass_phase_centers[best_pass_idx]
        half_width = best_local_half_width

        # time remaining to exit current pass
        exit_phase = wrap_phase(center + half_width)
        time_remaining = self._forward_phase_distance_s(orbit_phase, exit_phase)

        return GroundPassInfo(
            visible=True,
            pass_state=pass_state,
            elevation_deg=elevation_deg,
            link_quality=link_quality,
            time_to_next_pass_s=0.0,
            time_remaining_in_pass_s=time_remaining,
        )

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
        if elevation_deg < 10.0:
            return LinkQuality.POOR
        if elevation_deg < 25.0:
            return LinkQuality.MARGINAL
        if elevation_deg < 55.0:
            return LinkQuality.GOOD
        return LinkQuality.EXCELLENT

    # -----------------------------------------------------
    # Target opportunity model
    # -----------------------------------------------------

    def _compute_target_info(
        self,
        orbit_phase: float,
        sunlit_state: SunlightState,
        sun_elevation_deg: float,
    ) -> TargetInfo:
        """
        Target opportunity is based on:
        - phase being within a target window
        - daylight adequacy
        - small stochastic variability for realism
        """
        inside_window = False
        nearest_progress = None

        for center, half_width in zip(
            self.config.target_phase_centers,
            self.config.target_half_widths,
        ):
            dist = circular_phase_distance(orbit_phase, center)
            if dist <= half_width:
                inside_window = True
                progress = dist / max(half_width, 1e-9)
                nearest_progress = progress if nearest_progress is None else min(nearest_progress, progress)

        if not inside_window:
            return TargetInfo(
                over_target=False,
                target_opportunity=TargetOpportunity.NONE,
                quality_score=0.0,
            )

        # Opportunity strength peaks near window center
        geometric_quality = 1.0 - (nearest_progress if nearest_progress is not None else 1.0)
        geometric_quality = clamp(geometric_quality, 0.0, 1.0)

        # If eclipse or insufficient sun elevation, mark poor or none.
        if sunlit_state == SunlightState.ECLIPSE:
            return TargetInfo(
                over_target=False,
                target_opportunity=TargetOpportunity.NONE,
                quality_score=0.0,
            )

        if sun_elevation_deg < self.config.min_sun_elevation_for_imaging_deg:
            # We still expose a weak target signal for future planning,
            # but it should not count as a valid imaging opportunity.
            return TargetInfo(
                over_target=True,
                target_opportunity=TargetOpportunity.POOR_LIGHT,
                quality_score=0.15 * geometric_quality,
            )

        # Stochastic visibility modifier
        visible_draw = self.rng.random()
        if visible_draw > self.config.base_target_visibility_prob:
            return TargetInfo(
                over_target=False,
                target_opportunity=TargetOpportunity.NONE,
                quality_score=0.0,
            )

        # High-value target chance near center of window is slightly stronger.
        high_value_boost = 0.5 + 0.5 * geometric_quality
        hv_prob = clamp(
            self.config.high_value_target_prob * high_value_boost,
            0.0,
            1.0,
        )

        poor_light_prob = self.config.poor_light_target_prob * (1.0 - geometric_quality)

        draw = self.rng.random()
        if draw < hv_prob:
            return TargetInfo(
                over_target=True,
                target_opportunity=TargetOpportunity.HIGH_VALUE,
                quality_score=0.8 + 0.2 * geometric_quality,
            )
        if draw < hv_prob + poor_light_prob:
            return TargetInfo(
                over_target=True,
                target_opportunity=TargetOpportunity.POOR_LIGHT,
                quality_score=0.25 + 0.15 * geometric_quality,
            )

        return TargetInfo(
            over_target=True,
            target_opportunity=TargetOpportunity.VALID,
            quality_score=0.45 + 0.35 * geometric_quality,
        )

    # -----------------------------------------------------
    # Window timing helpers
    # -----------------------------------------------------

    def _time_to_next_window_s(
        self,
        orbit_phase: float,
        centers: Sequence[float],
        half_widths: Sequence[float],
        return_zero_if_inside: bool,
    ) -> float:
        candidates_s: List[float] = []

        for center, half_width in zip(centers, half_widths):
            dist = circular_phase_distance(orbit_phase, center)
            inside = dist <= half_width

            if inside and return_zero_if_inside:
                return 0.0

            entry_phase = wrap_phase(center - half_width)
            t_entry = self._forward_phase_distance_s(orbit_phase, entry_phase)
            candidates_s.append(t_entry)

        return min(candidates_s) if candidates_s else self.config.orbit_period_s

    def _forward_phase_distance_s(self, current_phase: float, target_phase: float) -> float:
        delta_phase = (wrap_phase(target_phase) - wrap_phase(current_phase)) % 1.0
        return delta_phase * self.config.orbit_period_s


# =========================================================
# Functional helpers
# =========================================================

def update_orbit_state(
    orbit_state: OrbitState,
    dt_s: float,
    config: OrbitConfig = DEFAULT_ORBIT_CONFIG,
    orbit_index: Optional[int] = None,
) -> OrbitState:
    """
    Functional convenience wrapper.
    """
    propagator = OrbitPropagator(config=config)
    return propagator.step(
        orbit_state=orbit_state,
        dt_s=dt_s,
        time_orbit_index=orbit_index,
    )


def initialize_orbit_state(
    orbit_state: OrbitState,
    config: OrbitConfig = DEFAULT_ORBIT_CONFIG,
    initial_phase: float = 0.0,
    orbit_index: int = 0,
) -> OrbitState:
    """
    Functional convenience wrapper.
    """
    propagator = OrbitPropagator(config=config)
    return propagator.reset(
        orbit_state=orbit_state,
        initial_phase=initial_phase,
        orbit_index=orbit_index,
    )


# =========================================================
# Example self-test helper
# =========================================================

def orbit_smoke_summary(
    config: OrbitConfig = DEFAULT_ORBIT_CONFIG,
    dt_s: float = 5.0,
    steps: int = 200,
) -> Dict[str, float]:
    """
    Small utility for quick subsystem sanity checks.
    """
    propagator = OrbitPropagator(config=config)
    state = OrbitState()
    propagator.reset(state, initial_phase=0.0, orbit_index=0)

    sunlit_steps = 0
    target_steps = 0
    pass_steps = 0
    max_sun_elev = -1e9
    max_pass_elev = 0.0

    orbit_index = 0
    prev_phase = state.orbit_phase

    for _ in range(steps):
        propagator.step(state, dt_s=dt_s, time_orbit_index=orbit_index)

        if state.orbit_phase < prev_phase:
            orbit_index += 1
        prev_phase = state.orbit_phase

        if state.is_sunlit:
            sunlit_steps += 1
        if state.over_target:
            target_steps += 1

        pass_info = propagator.get_ground_pass_info(state)
        if pass_info.visible:
            pass_steps += 1
            max_pass_elev = max(max_pass_elev, pass_info.elevation_deg)

        max_sun_elev = max(max_sun_elev, state.sun_elevation_deg)

    return {
        "sunlit_fraction_estimate": sunlit_steps / max(steps, 1),
        "target_fraction_estimate": target_steps / max(steps, 1),
        "pass_fraction_estimate": pass_steps / max(steps, 1),
        "max_sun_elevation_deg": max_sun_elev,
        "max_pass_elevation_deg": max_pass_elev,
        "final_orbit_phase": state.orbit_phase,
    }