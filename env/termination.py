from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models.enums import EpisodeEndReason, FaultLevel, FaultType
from models.state_models import SpacecraftState


# =========================================================
# Termination configuration
# =========================================================

@dataclass(frozen=True)
class TerminationConfig:
    """
    Episode termination policy for the CubeSat simulator.

    Philosophy:
    - terminate on hard mission-ending failures
    - optionally terminate on prolonged unsafe operation
    - support normal mission completion by max steps / max orbits
    """

    # -----------------------------------------------------
    # Nominal episode limits
    # -----------------------------------------------------
    max_steps: int = 8640              # e.g. 12 hours at dt = 5 s
    max_orbits: Optional[int] = None   # if set, terminate after this many completed orbits

    # -----------------------------------------------------
    # Battery / power survival limits
    # -----------------------------------------------------
    terminate_on_battery_depleted: bool = True
    battery_depleted_threshold_pct: float = 0.5

    terminate_on_brownout_risk: bool = False

    terminate_on_sustained_critical_battery: bool = True
    critical_battery_threshold_pct: float = 5.0
    critical_battery_max_consecutive_steps: int = 24   # ~2 min at dt=5 s

    # -----------------------------------------------------
    # Thermal survival limits
    # -----------------------------------------------------
    terminate_on_thermal_violation: bool = False

    terminate_on_sustained_battery_thermal_violation: bool = True
    battery_temp_too_cold_c: float = -2.0
    battery_temp_too_hot_c: float = 42.0
    battery_thermal_max_consecutive_steps: int = 24

    terminate_on_sustained_payload_thermal_violation: bool = True
    payload_temp_too_cold_c: float = -8.0
    payload_temp_too_hot_c: float = 45.0
    payload_thermal_max_consecutive_steps: int = 24

    terminate_on_sustained_bus_thermal_violation: bool = True
    bus_temp_too_cold_c: float = -15.0
    bus_temp_too_hot_c: float = 55.0
    bus_thermal_max_consecutive_steps: int = 24

    # -----------------------------------------------------
    # Fault policy
    # -----------------------------------------------------
    terminate_on_fatal_fault: bool = True
    terminate_on_any_critical_fault: bool = False

    terminate_on_specific_faults: Tuple[FaultType, ...] = (
        FaultType.CDH_RESET_LOOP,
        FaultType.STORAGE_CORRUPTION,
    )

    # -----------------------------------------------------
    # Data / avionics / storage policy
    # -----------------------------------------------------
    terminate_on_storage_failure: bool = True
    terminate_on_memory_overflow: bool = False
    memory_overflow_threshold_frac: float = 1.05

    terminate_on_filesystem_corruption: bool = True

    # -----------------------------------------------------
    # Optional "stuck mission" conditions
    # -----------------------------------------------------
    terminate_if_not_alive_flag_false: bool = True
    terminate_if_done_flag_true: bool = True

    # -----------------------------------------------------
    # Safety margin for manual aborts / external stop
    # -----------------------------------------------------
    terminate_on_manual_abort_flag: bool = True


DEFAULT_TERMINATION_CONFIG = TerminationConfig()


# =========================================================
# Persistent termination tracker
# =========================================================

@dataclass
class TerminationTracker:
    """
    Tracks consecutive unsafe-condition durations across steps.
    This avoids ending an episode due to one noisy or transient reading.
    """
    consecutive_critical_battery_steps: int = 0
    consecutive_battery_thermal_violation_steps: int = 0
    consecutive_payload_thermal_violation_steps: int = 0
    consecutive_bus_thermal_violation_steps: int = 0

    def reset(self) -> None:
        self.consecutive_critical_battery_steps = 0
        self.consecutive_battery_thermal_violation_steps = 0
        self.consecutive_payload_thermal_violation_steps = 0
        self.consecutive_bus_thermal_violation_steps = 0


# =========================================================
# Termination result
# =========================================================

@dataclass
class TerminationResult:
    """
    Rich result of evaluating episode termination.
    """
    done: bool
    reason: EpisodeEndReason = EpisodeEndReason.NOT_DONE
    reasons: List[str] = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []

    def add_reason(self, msg: str) -> None:
        self.reasons.append(msg)


# =========================================================
# Termination engine
# =========================================================

class TerminationEngine:
    """
    Evaluates whether the current episode should end.

    Recommended usage:
        tracker = TerminationTracker()
        engine = TerminationEngine()
        result = engine.evaluate(state, tracker)
    """

    def __init__(self, config: TerminationConfig = DEFAULT_TERMINATION_CONFIG):
        self.config = config

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def evaluate(
        self,
        state: SpacecraftState,
        tracker: TerminationTracker,
        manual_abort: bool = False,
    ) -> TerminationResult:
        """
        Evaluate the current spacecraft state and determine if the episode must terminate.
        """
        self._update_counters(state, tracker)

        # Ordered by priority: the first hard mission-ending condition wins.
        # That gives a single canonical EpisodeEndReason, while still returning
        # all applicable human-readable reasons in the result.
        checks = [
            self._check_manual_abort,
            self._check_external_flags,
            self._check_nominal_limits,
            self._check_battery_failure,
            self._check_thermal_failure,
            self._check_fault_failure,
            self._check_storage_failure,
        ]

        first_reason = EpisodeEndReason.NOT_DONE
        collected_messages: List[str] = []

        for check in checks:
            result = check(state, tracker, manual_abort)
            if result.done:
                if first_reason == EpisodeEndReason.NOT_DONE:
                    first_reason = result.reason
                collected_messages.extend(result.reasons)

        if first_reason != EpisodeEndReason.NOT_DONE:
            return TerminationResult(
                done=True,
                reason=first_reason,
                reasons=collected_messages,
            )

        return TerminationResult(done=False)

    # -----------------------------------------------------
    # Counter maintenance
    # -----------------------------------------------------

    def _update_counters(self, state: SpacecraftState, tracker: TerminationTracker) -> None:
        # Critical battery duration
        if state.eps.battery_soc_pct <= self.config.critical_battery_threshold_pct:
            tracker.consecutive_critical_battery_steps += 1
        else:
            tracker.consecutive_critical_battery_steps = 0

        # Battery thermal duration
        if (
            state.thermal.battery_temp_c < self.config.battery_temp_too_cold_c
            or state.thermal.battery_temp_c > self.config.battery_temp_too_hot_c
        ):
            tracker.consecutive_battery_thermal_violation_steps += 1
        else:
            tracker.consecutive_battery_thermal_violation_steps = 0

        # Payload thermal duration
        if (
            state.thermal.payload_temp_c < self.config.payload_temp_too_cold_c
            or state.thermal.payload_temp_c > self.config.payload_temp_too_hot_c
        ):
            tracker.consecutive_payload_thermal_violation_steps += 1
        else:
            tracker.consecutive_payload_thermal_violation_steps = 0

        # Bus thermal duration
        if (
            state.thermal.bus_temp_c < self.config.bus_temp_too_cold_c
            or state.thermal.bus_temp_c > self.config.bus_temp_too_hot_c
        ):
            tracker.consecutive_bus_thermal_violation_steps += 1
        else:
            tracker.consecutive_bus_thermal_violation_steps = 0

    # -----------------------------------------------------
    # Termination checks
    # -----------------------------------------------------

    def _check_manual_abort(
        self,
        state: SpacecraftState,
        tracker: TerminationTracker,
        manual_abort: bool,
    ) -> TerminationResult:
        result = TerminationResult(done=False)

        if self.config.terminate_on_manual_abort_flag and manual_abort:
            result.done = True
            result.reason = EpisodeEndReason.MANUAL_ABORT
            result.add_reason("Manual abort requested.")
        return result

    def _check_external_flags(
        self,
        state: SpacecraftState,
        tracker: TerminationTracker,
        manual_abort: bool,
    ) -> TerminationResult:
        result = TerminationResult(done=False)

        if self.config.terminate_if_done_flag_true and state.done:
            result.done = True
            result.reason = state.end_reason if state.end_reason != EpisodeEndReason.NOT_DONE else EpisodeEndReason.MANUAL_ABORT
            result.add_reason(f"State.done flag is already true with end reason {result.reason.name}.")

        if self.config.terminate_if_not_alive_flag_false and not state.alive:
            # Do not override a more specific existing reason if already set above
            if not result.done:
                result.done = True
                result.reason = state.end_reason if state.end_reason != EpisodeEndReason.NOT_DONE else EpisodeEndReason.FATAL_FAULT
            result.add_reason("Spacecraft alive flag is false.")

        return result

    def _check_nominal_limits(
        self,
        state: SpacecraftState,
        tracker: TerminationTracker,
        manual_abort: bool,
    ) -> TerminationResult:
        result = TerminationResult(done=False)

        if state.time.step_count >= self.config.max_steps:
            result.done = True
            result.reason = EpisodeEndReason.MAX_STEPS
            result.add_reason(
                f"Maximum step limit reached: {state.time.step_count} >= {self.config.max_steps}."
            )

        if self.config.max_orbits is not None and state.time.orbit_index >= self.config.max_orbits:
            if not result.done:
                result.done = True
                result.reason = EpisodeEndReason.MAX_STEPS
            result.add_reason(
                f"Maximum orbit limit reached: {state.time.orbit_index} >= {self.config.max_orbits}."
            )

        return result

    def _check_battery_failure(
        self,
        state: SpacecraftState,
        tracker: TerminationTracker,
        manual_abort: bool,
    ) -> TerminationResult:
        result = TerminationResult(done=False)

        if (
            self.config.terminate_on_battery_depleted
            and state.eps.battery_soc_pct <= self.config.battery_depleted_threshold_pct
        ):
            result.done = True
            result.reason = EpisodeEndReason.BATTERY_DEPLETED
            result.add_reason(
                f"Battery depleted: SoC {state.eps.battery_soc_pct:.2f}% <= "
                f"{self.config.battery_depleted_threshold_pct:.2f}%."
            )

        if self.config.terminate_on_brownout_risk and state.eps.brownout_risk:
            if not result.done:
                result.done = True
                result.reason = EpisodeEndReason.BATTERY_DEPLETED
            result.add_reason("Brownout risk flagged by EPS.")

        if (
            self.config.terminate_on_sustained_critical_battery
            and tracker.consecutive_critical_battery_steps >= self.config.critical_battery_max_consecutive_steps
        ):
            if not result.done:
                result.done = True
                result.reason = EpisodeEndReason.BATTERY_DEPLETED
            result.add_reason(
                "Battery remained in critical region for too long: "
                f"{tracker.consecutive_critical_battery_steps} consecutive steps "
                f">= {self.config.critical_battery_max_consecutive_steps}."
            )

        return result

    def _check_thermal_failure(
        self,
        state: SpacecraftState,
        tracker: TerminationTracker,
        manual_abort: bool,
    ) -> TerminationResult:
        result = TerminationResult(done=False)

        if self.config.terminate_on_thermal_violation and state.thermal.thermal_violation:
            result.done = True
            result.reason = EpisodeEndReason.THERMAL_FAILURE
            result.add_reason("General thermal violation flag is active.")

        if (
            self.config.terminate_on_sustained_battery_thermal_violation
            and tracker.consecutive_battery_thermal_violation_steps >= self.config.battery_thermal_max_consecutive_steps
        ):
            if not result.done:
                result.done = True
                result.reason = EpisodeEndReason.THERMAL_FAILURE
            result.add_reason(
                "Battery thermal violation sustained too long: "
                f"{tracker.consecutive_battery_thermal_violation_steps} consecutive steps "
                f">= {self.config.battery_thermal_max_consecutive_steps}. "
                f"Battery temp = {state.thermal.battery_temp_c:.2f} °C."
            )

        if (
            self.config.terminate_on_sustained_payload_thermal_violation
            and tracker.consecutive_payload_thermal_violation_steps >= self.config.payload_thermal_max_consecutive_steps
        ):
            if not result.done:
                result.done = True
                result.reason = EpisodeEndReason.THERMAL_FAILURE
            result.add_reason(
                "Payload thermal violation sustained too long: "
                f"{tracker.consecutive_payload_thermal_violation_steps} consecutive steps "
                f">= {self.config.payload_thermal_max_consecutive_steps}. "
                f"Payload temp = {state.thermal.payload_temp_c:.2f} °C."
            )

        if (
            self.config.terminate_on_sustained_bus_thermal_violation
            and tracker.consecutive_bus_thermal_violation_steps >= self.config.bus_thermal_max_consecutive_steps
        ):
            if not result.done:
                result.done = True
                result.reason = EpisodeEndReason.THERMAL_FAILURE
            result.add_reason(
                "Bus thermal violation sustained too long: "
                f"{tracker.consecutive_bus_thermal_violation_steps} consecutive steps "
                f">= {self.config.bus_thermal_max_consecutive_steps}. "
                f"Bus temp = {state.thermal.bus_temp_c:.2f} °C."
            )

        return result

    def _check_fault_failure(
        self,
        state: SpacecraftState,
        tracker: TerminationTracker,
        manual_abort: bool,
    ) -> TerminationResult:
        result = TerminationResult(done=False)

        if self.config.terminate_on_fatal_fault and state.faults.has_level_at_least(FaultLevel.FATAL):
            result.done = True
            result.reason = EpisodeEndReason.FATAL_FAULT
            result.add_reason("At least one active FATAL fault is present.")

        if self.config.terminate_on_any_critical_fault and state.faults.has_level_at_least(FaultLevel.CRITICAL):
            if not result.done:
                result.done = True
                result.reason = EpisodeEndReason.FATAL_FAULT
            result.add_reason("At least one active CRITICAL-or-higher fault is present.")

        active_fault_types = {
            fault.fault_type
            for fault in state.faults.active_faults
            if fault.active
        }

        matched = active_fault_types.intersection(set(self.config.terminate_on_specific_faults))
        if matched:
            if not result.done:
                result.done = True
                result.reason = EpisodeEndReason.FATAL_FAULT
            fault_names = ", ".join(sorted(f.name for f in matched))
            result.add_reason(f"Mission-ending fault type detected: {fault_names}.")

        return result

    def _check_storage_failure(
        self,
        state: SpacecraftState,
        tracker: TerminationTracker,
        manual_abort: bool,
    ) -> TerminationResult:
        result = TerminationResult(done=False)

        if self.config.terminate_on_storage_failure:
            if self.config.terminate_on_filesystem_corruption and (
                not state.cdh.filesystem_healthy or state.cdh.storage_corrupted
            ):
                result.done = True
                result.reason = EpisodeEndReason.STORAGE_FAILURE
                result.add_reason(
                    "Storage system unhealthy or corruption flag active."
                )

        if self.config.terminate_on_memory_overflow:
            utilization = (
                state.cdh.memory_used_mb / max(state.cdh.memory_capacity_mb, 1e-9)
            )
            if utilization >= self.config.memory_overflow_threshold_frac:
                if not result.done:
                    result.done = True
                    result.reason = EpisodeEndReason.STORAGE_FAILURE
                result.add_reason(
                    f"Memory overflow threshold crossed: utilization {utilization:.3f} "
                    f">= {self.config.memory_overflow_threshold_frac:.3f}."
                )

        return result


# =========================================================
# Convenience helpers
# =========================================================

def evaluate_termination(
    state: SpacecraftState,
    tracker: Optional[TerminationTracker] = None,
    config: TerminationConfig = DEFAULT_TERMINATION_CONFIG,
    manual_abort: bool = False,
) -> TerminationResult:
    """
    Functional helper for environments that do not want to manage an engine object.
    """
    engine = TerminationEngine(config=config)
    tracker = tracker or TerminationTracker()
    return engine.evaluate(
        state=state,
        tracker=tracker,
        manual_abort=manual_abort,
    )


def apply_termination_result(
    state: SpacecraftState,
    result: TerminationResult,
) -> None:
    """
    Applies a termination decision directly to the spacecraft state.
    """
    if result.done:
        state.done = True
        state.alive = False
        state.end_reason = result.reason


def reset_termination_tracker(tracker: TerminationTracker) -> None:
    tracker.reset()