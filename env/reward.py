from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.action_space import ActionEffectProfile, get_action_spec
from models.enums import (
    Action,
    EpisodeEndReason,
    FaultLevel,
    FaultType,
    FrameClass,
    RewardEvent,
    SpacecraftMode,
    SubsystemName,
)
from models.state_models import RewardBreakdown, SpacecraftState


# =========================================================
# Reward configuration
# =========================================================

@dataclass(frozen=True)
class RewardConfig:
    """
    Mission-centric reward shaping configuration.

    Philosophy:
    - reward delivered mission value, not button presses
    - heavily penalize spacecraft death / severe violations
    - positively shape good operational posture
    - keep per-step shaping small relative to mission events
    """

    # -----------------------------------------------------
    # Positive mission outcomes
    # -----------------------------------------------------
    reward_useful_image_captured: float = 3.0
    reward_high_value_image_captured: float = 5.0
    reward_classifier_correct_clear: float = 2.0
    reward_classifier_correct_cloudy: float = 1.5
    reward_store_clear_frame: float = 3.0
    reward_store_high_value_frame: float = 5.0
    reward_discard_cloudy_frame: float = 2.5
    reward_successful_low_rate_downlink_per_mb: float = 0.35
    reward_successful_high_rate_downlink_per_mb: float = 0.55
    reward_useful_data_downlinked_bonus: float = 8.0
    reward_fault_recovered: float = 4.0
    reward_enter_safe_mode_when_critical: float = 2.5

    # -----------------------------------------------------
    # Good housekeeping / posture shaping
    # -----------------------------------------------------
    reward_power_positive_step: float = 0.10
    reward_battery_healthy_step: float = 0.05
    reward_thermal_nominal_step: float = 0.05
    reward_sun_point_when_low_battery: float = 0.30
    reward_prepare_downlink_when_pass_and_queue: float = 0.50
    reward_desaturate_when_needed: float = 0.40
    reward_hold_margin_before_eclipse: float = 0.20

    # -----------------------------------------------------
    # Waste / suboptimal action penalties
    # -----------------------------------------------------
    penalty_invalid_action: float = -6.0
    penalty_capture_without_valid_target: float = -4.0
    penalty_capture_under_bad_pointing: float = -4.5
    penalty_capture_when_power_low: float = -3.5
    penalty_store_cloudy_frame: float = -3.0
    penalty_discard_clear_frame: float = -4.0
    penalty_discard_high_value_frame: float = -6.0
    penalty_downlink_with_empty_queue: float = -5.0
    penalty_downlink_outside_pass: float = -6.0
    penalty_prepare_downlink_without_visibility: float = -2.5
    penalty_unnecessary_heater_use: float = -1.5
    penalty_unnecessary_safe_mode: float = -1.0
    penalty_desaturate_unnecessarily: float = -1.2
    penalty_no_op_when_urgent_queue_and_pass: float = -1.5
    penalty_no_op_when_critical_fault: float = -2.5
    penalty_running_payload_in_bad_thermal: float = -3.0
    penalty_payload_warmup_unnecessary: float = -1.0

    # -----------------------------------------------------
    # Resource / health penalties
    # -----------------------------------------------------
    penalty_power_negative_step: float = -0.20
    penalty_low_battery_step: float = -0.25
    penalty_critical_battery_step: float = -0.75
    penalty_memory_high_step: float = -0.10
    penalty_memory_critical_step: float = -0.40
    penalty_thermal_warning_step: float = -0.25
    penalty_thermal_violation_step: float = -1.00
    penalty_link_poor_for_high_rate: float = -2.0
    penalty_wheel_saturation_step: float = -0.15

    # -----------------------------------------------------
    # Fault penalties
    # -----------------------------------------------------
    penalty_warning_fault: float = -0.75
    penalty_limit_fault: float = -2.0
    penalty_critical_fault: float = -5.0
    penalty_fatal_fault: float = -15.0

    # -----------------------------------------------------
    # Terminal penalties / bonuses
    # -----------------------------------------------------
    terminal_bonus_survive_episode: float = 5.0
    terminal_penalty_battery_depleted: float = -25.0
    terminal_penalty_thermal_failure: float = -25.0
    terminal_penalty_fatal_fault: float = -30.0
    terminal_penalty_storage_failure: float = -18.0
    terminal_penalty_manual_abort: float = -8.0

    # -----------------------------------------------------
    # Anti-exploit guardrails
    # -----------------------------------------------------
    max_abs_total_reward_per_step: float = 25.0


DEFAULT_REWARD_CONFIG = RewardConfig()


# =========================================================
# Transition event structure
# =========================================================

@dataclass
class TransitionContext:
    """
    All transition-side information needed for reward computation.

    This separates reward logic from the simulator implementation.
    The environment can populate only the fields it knows; the reward
    engine handles missing info gracefully.
    """
    action: Action
    action_valid: bool = True
    action_profile: Optional[ActionEffectProfile] = None

    # Frame / classifier outcomes
    capture_attempted: bool = False
    capture_succeeded: bool = False
    classifier_ran: bool = False
    classifier_correct: Optional[bool] = None

    frame_was_present_before: bool = False
    frame_was_useful_before: bool = False
    frame_was_high_value_before: bool = False
    frame_was_cloudy_before: bool = False
    frame_stored: bool = False
    frame_discarded: bool = False

    # Downlink outcomes
    downlink_attempted: bool = False
    downlink_succeeded: bool = False
    useful_data_downlinked_mb: float = 0.0
    total_data_downlinked_mb: float = 0.0

    # Fault / recovery outcomes
    entered_safe_mode_this_step: bool = False
    fault_recovered: bool = False
    new_faults_added: List[Tuple[SubsystemName, FaultType, FaultLevel]] = field(default_factory=list)

    # State deltas
    delta_battery_soc_pct: float = 0.0
    delta_memory_used_mb: float = 0.0
    delta_queue_mb: float = 0.0

    # Transition terminal status
    became_done_this_step: bool = False
    end_reason: EpisodeEndReason = EpisodeEndReason.NOT_DONE


# =========================================================
# Reward engine
# =========================================================

class RewardEngine:
    """
    Central reward function for the CubeSat simulator.

    Design goals:
    - transparent reward breakdown for every step
    - robust to partially populated transition context
    - easy to calibrate and test
    """

    def __init__(self, config: RewardConfig = DEFAULT_REWARD_CONFIG):
        self.config = config

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def compute_reward(
        self,
        prev_state: SpacecraftState,
        next_state: SpacecraftState,
        ctx: TransitionContext,
    ) -> RewardBreakdown:
        rb = RewardBreakdown()

        self._reward_action_validity(rb, prev_state, next_state, ctx)
        self._reward_imaging_pipeline(rb, prev_state, next_state, ctx)
        self._reward_downlink_pipeline(rb, prev_state, next_state, ctx)
        self._reward_housekeeping(rb, prev_state, next_state, ctx)
        self._reward_resource_health(rb, prev_state, next_state, ctx)
        self._reward_faults_and_recovery(rb, prev_state, next_state, ctx)
        self._reward_terminal(rb, prev_state, next_state, ctx)

        self._clip_total_reward(rb)
        return rb

    # -----------------------------------------------------
    # 1) Action validity and intent
    # -----------------------------------------------------

    def _reward_action_validity(
        self,
        rb: RewardBreakdown,
        prev_state: SpacecraftState,
        next_state: SpacecraftState,
        ctx: TransitionContext,
    ) -> None:
        if not ctx.action_valid:
            rb.add(
                "invalid_action",
                self.config.penalty_invalid_action,
            )
            return

        action = ctx.action

        if action == Action.ENTER_SAFE_MODE:
            if self._is_critical_situation(prev_state):
                rb.add(
                    "safe_mode_entry_critical",
                    self.config.reward_enter_safe_mode_when_critical,
                    RewardEvent.SAFE_MODE_ENTRY,
                )
            else:
                rb.add(
                    "safe_mode_entry_unnecessary",
                    self.config.penalty_unnecessary_safe_mode,
                )

        if action == Action.NO_OP:
            if prev_state.comms.gs_visible and prev_state.cdh.downlink_queue_mb > 0.0:
                rb.add(
                    "noop_during_usable_pass",
                    self.config.penalty_no_op_when_urgent_queue_and_pass,
                )
            if prev_state.critical_fault:
                rb.add(
                    "noop_critical_fault",
                    self.config.penalty_no_op_when_critical_fault,
                )

        if action == Action.PREPARE_DOWNLINK:
            if prev_state.comms.gs_visible and prev_state.cdh.downlink_queue_mb > 0.0:
                rb.add(
                    "prepare_downlink_good_timing",
                    self.config.reward_prepare_downlink_when_pass_and_queue,
                )
            else:
                rb.add(
                    "prepare_downlink_bad_timing",
                    self.config.penalty_prepare_downlink_without_visibility,
                )

        if action == Action.DESATURATE_WHEELS:
            if prev_state.adcs.wheels_saturated:
                rb.add(
                    "desaturate_when_needed",
                    self.config.reward_desaturate_when_needed,
                )
            else:
                rb.add(
                    "desaturate_unnecessary",
                    self.config.penalty_desaturate_unnecessarily,
                )

        if action == Action.SUN_POINT_CHARGE:
            if prev_state.eps.battery_soc_pct < 30.0 and prev_state.orbit.is_sunlit:
                rb.add(
                    "sun_point_when_low_battery",
                    self.config.reward_sun_point_when_low_battery,
                )

        if action == Action.PAYLOAD_WARMUP:
            if prev_state.payload.mode.name in ("READY", "IMAGING"):
                rb.add(
                    "warmup_unnecessary",
                    self.config.penalty_payload_warmup_unnecessary,
                )

    # -----------------------------------------------------
    # 2) Imaging / classifier / storage pipeline
    # -----------------------------------------------------

    def _reward_imaging_pipeline(
        self,
        rb: RewardBreakdown,
        prev_state: SpacecraftState,
        next_state: SpacecraftState,
        ctx: TransitionContext,
    ) -> None:
        action = ctx.action

        # Capture attempt quality
        if action == Action.CAPTURE_IMAGE or ctx.capture_attempted:
            if not prev_state.orbit.over_target or not prev_state.orbit.day_imaging_valid:
                rb.add(
                    "capture_without_valid_target",
                    self.config.penalty_capture_without_valid_target,
                )

            if prev_state.adcs.pointing_quality.value < 2:
                rb.add(
                    "capture_bad_pointing",
                    self.config.penalty_capture_under_bad_pointing,
                )

            if prev_state.eps.battery_soc_pct < 20.0:
                rb.add(
                    "capture_low_power",
                    self.config.penalty_capture_when_power_low,
                )

            if prev_state.thermal.payload_temp_band.value in (0, 4):
                rb.add(
                    "capture_bad_payload_thermal",
                    self.config.penalty_running_payload_in_bad_thermal,
                )

        # Successful capture reward
        if ctx.capture_succeeded:
            frame_class = next_state.payload.current_frame_class

            if frame_class == FrameClass.HIGH_VALUE_CLEAR:
                rb.add(
                    "capture_high_value_frame",
                    self.config.reward_high_value_image_captured,
                    RewardEvent.USEFUL_IMAGE_CAPTURED,
                )
            elif frame_class in (FrameClass.CLEAR,):
                rb.add(
                    "capture_useful_frame",
                    self.config.reward_useful_image_captured,
                    RewardEvent.USEFUL_IMAGE_CAPTURED,
                )
            elif frame_class == FrameClass.CLOUDY:
                rb.add(
                    "capture_cloudy_frame",
                    0.0,
                    RewardEvent.CLOUDY_IMAGE_CAPTURED,
                )

        # Classifier reward
        if ctx.classifier_ran and ctx.classifier_correct is not None:
            if ctx.classifier_correct:
                if next_state.payload.current_frame_class in (FrameClass.CLEAR, FrameClass.HIGH_VALUE_CLEAR):
                    rb.add(
                        "classifier_correct_clear",
                        self.config.reward_classifier_correct_clear,
                        RewardEvent.CLASSIFICATION_CORRECT,
                    )
                elif next_state.payload.current_frame_class == FrameClass.CLOUDY:
                    rb.add(
                        "classifier_correct_cloudy",
                        self.config.reward_classifier_correct_cloudy,
                        RewardEvent.CLASSIFICATION_CORRECT,
                    )
            else:
                rb.add(
                    "classifier_wrong",
                    -abs(min(
                        self.config.reward_classifier_correct_clear,
                        self.config.reward_classifier_correct_cloudy,
                    )),
                    RewardEvent.CLASSIFICATION_WRONG,
                )

        # Store / discard logic
        if ctx.frame_stored:
            if ctx.frame_was_high_value_before:
                rb.add(
                    "store_high_value_frame",
                    self.config.reward_store_high_value_frame,
                    RewardEvent.FRAME_STORED,
                )
            elif ctx.frame_was_useful_before:
                rb.add(
                    "store_useful_frame",
                    self.config.reward_store_clear_frame,
                    RewardEvent.FRAME_STORED,
                )
            elif ctx.frame_was_cloudy_before:
                rb.add(
                    "store_cloudy_frame",
                    self.config.penalty_store_cloudy_frame,
                    RewardEvent.FRAME_STORED,
                )

        if ctx.frame_discarded:
            if ctx.frame_was_cloudy_before:
                rb.add(
                    "discard_cloudy_frame",
                    self.config.reward_discard_cloudy_frame,
                    RewardEvent.FRAME_DISCARDED,
                )
            elif ctx.frame_was_high_value_before:
                rb.add(
                    "discard_high_value_frame",
                    self.config.penalty_discard_high_value_frame,
                    RewardEvent.FRAME_DISCARDED,
                )
            elif ctx.frame_was_useful_before:
                rb.add(
                    "discard_clear_frame",
                    self.config.penalty_discard_clear_frame,
                    RewardEvent.FRAME_DISCARDED,
                )

    # -----------------------------------------------------
    # 3) Downlink pipeline
    # -----------------------------------------------------

    def _reward_downlink_pipeline(
        self,
        rb: RewardBreakdown,
        prev_state: SpacecraftState,
        next_state: SpacecraftState,
        ctx: TransitionContext,
    ) -> None:
        action = ctx.action

        if action in (Action.DOWNLINK_LOW_RATE, Action.DOWNLINK_HIGH_RATE) or ctx.downlink_attempted:
            if prev_state.cdh.downlink_queue_mb <= 0.0:
                rb.add(
                    "downlink_empty_queue",
                    self.config.penalty_downlink_with_empty_queue,
                )

            if not prev_state.comms.gs_visible:
                rb.add(
                    "downlink_outside_pass",
                    self.config.penalty_downlink_outside_pass,
                    RewardEvent.FAILED_DOWNLINK,
                )

            if action == Action.DOWNLINK_HIGH_RATE and prev_state.comms.link_quality.value < 3:
                rb.add(
                    "high_rate_poor_link",
                    self.config.penalty_link_poor_for_high_rate,
                )

        if ctx.downlink_succeeded and ctx.total_data_downlinked_mb > 0.0:
            per_mb = (
                self.config.reward_successful_high_rate_downlink_per_mb
                if action == Action.DOWNLINK_HIGH_RATE
                else self.config.reward_successful_low_rate_downlink_per_mb
            )

            rb.add(
                "data_downlinked_mb",
                per_mb * ctx.total_data_downlinked_mb,
                RewardEvent.SUCCESSFUL_DOWNLINK,
            )

            if ctx.useful_data_downlinked_mb > 0.0:
                rb.add(
                    "useful_data_downlinked_bonus",
                    self.config.reward_useful_data_downlinked_bonus,
                    RewardEvent.SUCCESSFUL_DOWNLINK,
                )

    # -----------------------------------------------------
    # 4) Housekeeping / posture shaping
    # -----------------------------------------------------

    def _reward_housekeeping(
        self,
        rb: RewardBreakdown,
        prev_state: SpacecraftState,
        next_state: SpacecraftState,
        ctx: TransitionContext,
    ) -> None:
        # Power-positive operation
        if next_state.power_positive:
            rb.add(
                "power_positive_step",
                self.config.reward_power_positive_step,
                RewardEvent.POWER_POSITIVE_STEP,
            )
        else:
            rb.add(
                "power_negative_step",
                self.config.penalty_power_negative_step,
                RewardEvent.POWER_NEGATIVE_STEP,
            )

        # Battery in healthy region
        if 35.0 <= next_state.eps.battery_soc_pct <= 90.0:
            rb.add(
                "battery_healthy_step",
                self.config.reward_battery_healthy_step,
            )

        # Thermal nominal
        if (
            next_state.thermal.battery_temp_band.name == "NOMINAL"
            and next_state.thermal.payload_temp_band.name == "NOMINAL"
        ):
            rb.add(
                "thermal_nominal_step",
                self.config.reward_thermal_nominal_step,
            )

        # Avoid entering eclipse with too little battery
        if prev_state.orbit.is_sunlit and next_state.orbit.in_eclipse and next_state.eps.battery_soc_pct >= 35.0:
            rb.add(
                "hold_margin_before_eclipse",
                self.config.reward_hold_margin_before_eclipse,
            )

        # Heater misuse
        if ctx.action in (Action.ENABLE_BATTERY_HEATER, Action.ENABLE_PAYLOAD_HEATER):
            if ctx.action == Action.ENABLE_BATTERY_HEATER and prev_state.thermal.battery_temp_band.name in ("NOMINAL", "WARM", "TOO_HOT"):
                rb.add(
                    "battery_heater_unnecessary",
                    self.config.penalty_unnecessary_heater_use,
                )
            if ctx.action == Action.ENABLE_PAYLOAD_HEATER and prev_state.thermal.payload_temp_band.name in ("NOMINAL", "WARM", "TOO_HOT"):
                rb.add(
                    "payload_heater_unnecessary",
                    self.config.penalty_unnecessary_heater_use,
                )

    # -----------------------------------------------------
    # 5) Resource and subsystem health
    # -----------------------------------------------------

    def _reward_resource_health(
        self,
        rb: RewardBreakdown,
        prev_state: SpacecraftState,
        next_state: SpacecraftState,
        ctx: TransitionContext,
    ) -> None:
        # Battery region penalties
        if next_state.eps.battery_soc_pct < 30.0:
            rb.add(
                "low_battery_step",
                self.config.penalty_low_battery_step,
            )
        if next_state.eps.battery_soc_pct < 10.0:
            rb.add(
                "critical_battery_step",
                self.config.penalty_critical_battery_step,
            )

        # Memory pressure penalties
        if next_state.cdh.memory_pressure.name == "HIGH":
            rb.add(
                "memory_high_step",
                self.config.penalty_memory_high_step,
            )
        elif next_state.cdh.memory_pressure.name == "CRITICAL":
            rb.add(
                "memory_critical_step",
                self.config.penalty_memory_critical_step,
                RewardEvent.MEMORY_OVERFLOW,
            )

        # Thermal penalties
        if next_state.thermal.battery_temp_band.name in ("COLD", "WARM") or next_state.thermal.payload_temp_band.name in ("COLD", "WARM"):
            rb.add(
                "thermal_warning_step",
                self.config.penalty_thermal_warning_step,
            )
        if next_state.thermal.thermal_violation:
            rb.add(
                "thermal_violation_step",
                self.config.penalty_thermal_violation_step,
                RewardEvent.THERMAL_VIOLATION,
            )

        # Wheel saturation
        if next_state.adcs.wheels_saturated:
            rb.add(
                "wheel_saturation_step",
                self.config.penalty_wheel_saturation_step,
            )

    # -----------------------------------------------------
    # 6) Faults and recovery
    # -----------------------------------------------------

    def _reward_faults_and_recovery(
        self,
        rb: RewardBreakdown,
        prev_state: SpacecraftState,
        next_state: SpacecraftState,
        ctx: TransitionContext,
    ) -> None:
        # New faults
        for _, fault_type, level in ctx.new_faults_added:
            if level == FaultLevel.WARNING:
                rb.add(
                    f"fault_{fault_type.name.lower()}_warning",
                    self.config.penalty_warning_fault,
                )
            elif level == FaultLevel.LIMIT:
                rb.add(
                    f"fault_{fault_type.name.lower()}_limit",
                    self.config.penalty_limit_fault,
                )
            elif level == FaultLevel.CRITICAL:
                rb.add(
                    f"fault_{fault_type.name.lower()}_critical",
                    self.config.penalty_critical_fault,
                )
            elif level == FaultLevel.FATAL:
                rb.add(
                    f"fault_{fault_type.name.lower()}_fatal",
                    self.config.penalty_fatal_fault,
                )

        # Recovery bonus
        if ctx.fault_recovered:
            rb.add(
                "fault_recovered",
                self.config.reward_fault_recovered,
                RewardEvent.FAULT_RECOVERED,
            )

    # -----------------------------------------------------
    # 7) Terminal rewards/penalties
    # -----------------------------------------------------

    def _reward_terminal(
        self,
        rb: RewardBreakdown,
        prev_state: SpacecraftState,
        next_state: SpacecraftState,
        ctx: TransitionContext,
    ) -> None:
        if not next_state.done and ctx.end_reason == EpisodeEndReason.NOT_DONE:
            return

        reason = ctx.end_reason if ctx.end_reason != EpisodeEndReason.NOT_DONE else next_state.end_reason

        if reason == EpisodeEndReason.MAX_STEPS:
            rb.add(
                "terminal_survive_episode",
                self.config.terminal_bonus_survive_episode,
            )
        elif reason == EpisodeEndReason.BATTERY_DEPLETED:
            rb.add(
                "terminal_battery_depleted",
                self.config.terminal_penalty_battery_depleted,
            )
        elif reason == EpisodeEndReason.THERMAL_FAILURE:
            rb.add(
                "terminal_thermal_failure",
                self.config.terminal_penalty_thermal_failure,
            )
        elif reason == EpisodeEndReason.FATAL_FAULT:
            rb.add(
                "terminal_fatal_fault",
                self.config.terminal_penalty_fatal_fault,
            )
        elif reason == EpisodeEndReason.STORAGE_FAILURE:
            rb.add(
                "terminal_storage_failure",
                self.config.terminal_penalty_storage_failure,
            )
        elif reason == EpisodeEndReason.MANUAL_ABORT:
            rb.add(
                "terminal_manual_abort",
                self.config.terminal_penalty_manual_abort,
            )

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------

    def _is_critical_situation(self, state: SpacecraftState) -> bool:
        return any(
            [
                state.eps.battery_soc_pct < 12.0,
                state.thermal.thermal_violation,
                state.critical_fault,
                state.mode in (SpacecraftMode.SAFE, SpacecraftMode.SURVIVAL, SpacecraftMode.FAULT_RECOVERY),
            ]
        )

    def _clip_total_reward(self, rb: RewardBreakdown) -> None:
        """
        Keep one-step reward from exploding due to accidental double-counting.
        """
        max_abs = abs(self.config.max_abs_total_reward_per_step)
        if rb.total > max_abs:
            scale = max_abs / max(rb.total, 1e-9)
            self._rescale_breakdown(rb, scale)
        elif rb.total < -max_abs:
            scale = max_abs / max(abs(rb.total), 1e-9)
            self._rescale_breakdown(rb, scale)

    @staticmethod
    def _rescale_breakdown(rb: RewardBreakdown, scale: float) -> None:
        rb.total *= scale
        for key in list(rb.components.keys()):
            rb.components[key] *= scale


# =========================================================
# Convenience helpers for env integration
# =========================================================

def infer_transition_context(
    prev_state: SpacecraftState,
    next_state: SpacecraftState,
    action: Action,
    action_valid: bool = True,
) -> TransitionContext:
    """
    Best-effort automatic context inference from before/after states.

    The environment can use this directly, then optionally override fields
    with more exact knowledge from simulator internals.
    """
    prev_frame_class = prev_state.payload.current_frame_class
    next_frame_class = next_state.payload.current_frame_class

    prev_has_frame = prev_state.payload.has_frame
    next_has_frame = next_state.payload.has_frame

    prev_queue = prev_state.cdh.downlink_queue_mb
    next_queue = next_state.cdh.downlink_queue_mb

    prev_mem = prev_state.cdh.memory_used_mb
    next_mem = next_state.cdh.memory_used_mb

    prev_safe = prev_state.mode == SpacecraftMode.SAFE
    next_safe = next_state.mode == SpacecraftMode.SAFE

    # Frame semantics before action
    frame_was_cloudy_before = prev_frame_class == FrameClass.CLOUDY
    frame_was_useful_before = prev_frame_class in (FrameClass.CLEAR, FrameClass.HIGH_VALUE_CLEAR)
    frame_was_high_value_before = prev_frame_class == FrameClass.HIGH_VALUE_CLEAR

    # Heuristic event inference
    capture_attempted = action == Action.CAPTURE_IMAGE
    capture_succeeded = (
        action == Action.CAPTURE_IMAGE
        and (not prev_has_frame)
        and next_has_frame
    )

    classifier_ran = action == Action.RUN_CLASSIFIER
    # Cannot truly infer correctness without ground truth; leave None.
    classifier_correct = None

    frame_stored = (
        action == Action.STORE_FRAME
        and prev_has_frame
        and not next_has_frame
        and next_mem >= prev_mem
    )

    frame_discarded = (
        action == Action.DISCARD_FRAME
        and prev_has_frame
        and not next_has_frame
    )

    downlink_attempted = action in (Action.DOWNLINK_LOW_RATE, Action.DOWNLINK_HIGH_RATE)
    downlink_succeeded = downlink_attempted and next_queue < prev_queue
    total_data_downlinked_mb = max(0.0, prev_queue - next_queue)

    # Useful data estimation: if the queue shrank during downlink, treat that as useful by default
    # unless the environment later overrides with a truer value.
    useful_data_downlinked_mb = total_data_downlinked_mb

    entered_safe_mode_this_step = (not prev_safe) and next_safe

    # New faults inference via active fault diff
    prev_fault_keys = {
        (f.subsystem, f.fault_type, f.level)
        for f in prev_state.faults.active_faults
        if f.active
    }
    next_fault_keys = {
        (f.subsystem, f.fault_type, f.level)
        for f in next_state.faults.active_faults
        if f.active
    }
    new_faults_added = list(next_fault_keys - prev_fault_keys)

    fault_recovered = (
        len(next_fault_keys) < len(prev_fault_keys)
        and action in (
            Action.FAULT_RECOVERY,
            Action.RESET_PAYLOAD,
            Action.RESET_COMMS,
            Action.RESET_ADCS,
            Action.RESET_CDH,
        )
    )

    ctx = TransitionContext(
        action=action,
        action_valid=action_valid,
        action_profile=get_action_spec(action).profile(),
        capture_attempted=capture_attempted,
        capture_succeeded=capture_succeeded,
        classifier_ran=classifier_ran,
        classifier_correct=classifier_correct,
        frame_was_present_before=prev_has_frame,
        frame_was_useful_before=frame_was_useful_before,
        frame_was_high_value_before=frame_was_high_value_before,
        frame_was_cloudy_before=frame_was_cloudy_before,
        frame_stored=frame_stored,
        frame_discarded=frame_discarded,
        downlink_attempted=downlink_attempted,
        downlink_succeeded=downlink_succeeded,
        useful_data_downlinked_mb=useful_data_downlinked_mb,
        total_data_downlinked_mb=total_data_downlinked_mb,
        entered_safe_mode_this_step=entered_safe_mode_this_step,
        fault_recovered=fault_recovered,
        new_faults_added=new_faults_added,
        delta_battery_soc_pct=next_state.eps.battery_soc_pct - prev_state.eps.battery_soc_pct,
        delta_memory_used_mb=next_mem - prev_mem,
        delta_queue_mb=next_queue - prev_queue,
        became_done_this_step=(not prev_state.done) and next_state.done,
        end_reason=next_state.end_reason,
    )
    return ctx


def compute_step_reward(
    prev_state: SpacecraftState,
    next_state: SpacecraftState,
    action: Action,
    action_valid: bool = True,
    ctx: Optional[TransitionContext] = None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> RewardBreakdown:
    """
    One-shot helper for environments that want a simple functional API.
    """
    engine = RewardEngine(config=config)
    context = ctx if ctx is not None else infer_transition_context(
        prev_state=prev_state,
        next_state=next_state,
        action=action,
        action_valid=action_valid,
    )
    return engine.compute_reward(prev_state, next_state, context)