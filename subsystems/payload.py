from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional

from models.action_space import ActionEffectProfile
from models.enums import (
    Action,
    FrameClass,
    PayloadMode,
    PointingQuality,
    TargetOpportunity,
)
from models.state_models import SpacecraftState


# =========================================================
# Configuration
# =========================================================

@dataclass(frozen=True)
class PayloadConfig:
    """
    Operational payload model for an Earth-observation CubeSat.

    This version is intentionally classifier-independent for now.
    It simulates:
    - payload warmup / readiness
    - imaging eligibility
    - synthetic frame generation
    - synthetic cloud / usefulness labeling
    - confidence-like score generation
    - frame lifecycle before CDH storage/downlink handling

    It is designed so a real classifier can later replace only the
    synthetic frame-label generation part without changing the rest
    of the environment architecture.

    Units:
    - time: s
    - size: MB
    - probabilities / usefulness / confidence: [0,1]
    """

    # -----------------------------------------------------
    # Warmup / readiness timing
    # -----------------------------------------------------
    warmup_duration_s: float = 25.0
    capture_cooldown_s: float = 6.0

    # -----------------------------------------------------
    # Capture requirements
    # -----------------------------------------------------
    min_battery_soc_for_capture_pct: float = 18.0
    min_battery_soc_for_warmup_pct: float = 12.0
    max_payload_temp_for_capture_c: float = 40.0
    min_payload_temp_for_capture_c: float = -5.0

    require_daylight_for_capture: bool = True
    require_target_for_capture: bool = True
    require_good_pointing: bool = True
    require_settled_attitude: bool = True

    min_pointing_quality: PointingQuality = PointingQuality.USABLE

    # -----------------------------------------------------
    # Frame sizing
    # -----------------------------------------------------
    raw_frame_size_mb_mean: float = 32.0
    raw_frame_size_mb_jitter: float = 4.0
    min_frame_size_mb: float = 24.0
    max_frame_size_mb: float = 40.0

    # -----------------------------------------------------
    # Synthetic scene generation
    # -----------------------------------------------------
    # These are baseline probabilities. Final class depends on
    # target quality, sun elevation, pointing quality, and stochasticity.
    base_cloudy_prob: float = 0.30
    base_partly_cloudy_prob: float = 0.25
    base_clear_prob: float = 0.35
    base_high_value_clear_prob: float = 0.10

    # Opportunity-dependent adjustments
    high_value_bonus_if_high_value_target: float = 0.22
    clear_bonus_if_valid_target: float = 0.10
    cloudy_bonus_if_poor_light: float = 0.18

    # Illumination influence
    poor_light_penalty_on_usefulness: float = 0.22
    excellent_light_bonus_on_usefulness: float = 0.12

    # Pointing degradation
    coarse_pointing_cloud_penalty: float = 0.10
    coarse_pointing_usefulness_penalty: float = 0.15

    # -----------------------------------------------------
    # Confidence / label realism
    # -----------------------------------------------------
    min_confidence: float = 0.55
    max_confidence: float = 0.98

    # -----------------------------------------------------
    # Randomness
    # -----------------------------------------------------
    random_seed: Optional[int] = None

    # -----------------------------------------------------
    # Numerical guards
    # -----------------------------------------------------
    min_prob: float = 0.0
    max_prob: float = 1.0
    min_usefulness: float = 0.0
    max_usefulness: float = 1.0


DEFAULT_PAYLOAD_CONFIG = PayloadConfig()


# =========================================================
# Helpers
# =========================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# =========================================================
# Breakdown
# =========================================================

@dataclass(frozen=True)
class PayloadBreakdown:
    """
    Detailed payload update/capture report for debugging and testing.
    """
    mode: str
    payload_enabled: bool
    imaging_ready: bool
    warmup_time_remaining_s: float
    imaging_cooldown_s: float
    capture_attempted: bool
    capture_succeeded: bool
    frame_generated: bool
    current_frame_id: int
    current_frame_size_mb: float
    current_frame_class: str
    current_frame_cloud_prob: float
    current_frame_usefulness: float
    classifier_confidence: float


# =========================================================
# Payload subsystem
# =========================================================

class PayloadSubsystem:
    """
    Operational payload model.

    Responsibilities:
    - manage warmup and readiness
    - decide whether capture is physically/operationally allowed
    - generate synthetic frame content when capture succeeds
    - expose a clean frame representation for RL/CDH
    """

    def __init__(self, config: PayloadConfig = DEFAULT_PAYLOAD_CONFIG):
        self.config = config
        self.rng = random.Random(config.random_seed)
        self._frame_counter = 0

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def initialize(self, state: SpacecraftState) -> None:
        state.payload.payload_enabled = False
        state.payload.imaging_ready = False
        state.payload.imaging_cooldown_s = 0.0
        state.payload.warmup_time_remaining_s = 0.0
        state.payload.mode = PayloadMode.OFF
        state.payload.clear_current_frame()

    def step(
        self,
        state: SpacecraftState,
        dt_s: float,
        action: Optional[Action] = None,
        action_profile: Optional[ActionEffectProfile] = None,
    ) -> PayloadBreakdown:
        """
        Advance payload state by one simulation step.
        """
        capture_attempted = False
        capture_succeeded = False
        frame_generated = False

        # Passive timers
        self._update_timers(state, dt_s)

        # Action-driven transitions
        if action == Action.PAYLOAD_WARMUP:
            self._start_or_continue_warmup(state)

        elif action == Action.CAPTURE_IMAGE:
            capture_attempted = True
            capture_succeeded = self._attempt_capture(state)
            frame_generated = capture_succeeded

        elif action == Action.RUN_CLASSIFIER:
            real_frame = action_profile.real_frame if action_profile and hasattr(action_profile, 'real_frame') else None
            self._refresh_synthetic_inference_metadata(state, real_frame=real_frame)

        elif action == Action.RESET_PAYLOAD:
            self._reset_payload(state)

        # Maintain mode consistency if no explicit action was issued
        self._maintain_mode(state)

        return PayloadBreakdown(
            mode=state.payload.mode.name,
            payload_enabled=state.payload.payload_enabled,
            imaging_ready=state.payload.imaging_ready,
            warmup_time_remaining_s=state.payload.warmup_time_remaining_s,
            imaging_cooldown_s=state.payload.imaging_cooldown_s,
            capture_attempted=capture_attempted,
            capture_succeeded=capture_succeeded,
            frame_generated=frame_generated,
            current_frame_id=state.payload.current_frame_id or -1,
            current_frame_size_mb=state.payload.current_frame_size_mb,
            current_frame_class=state.payload.current_frame_class.name,
            current_frame_cloud_prob=state.payload.current_frame_cloud_prob,
            current_frame_usefulness=state.payload.current_frame_usefulness,
            classifier_confidence=state.payload.classifier_confidence,
        )

    # -----------------------------------------------------
    # Timer / mode handling
    # -----------------------------------------------------

    def _update_timers(self, state: SpacecraftState, dt_s: float) -> None:
        if state.payload.warmup_time_remaining_s > 0.0:
            state.payload.warmup_time_remaining_s = max(
                0.0,
                state.payload.warmup_time_remaining_s - dt_s,
            )
            if state.payload.warmup_time_remaining_s <= 0.0:
                state.payload.payload_enabled = True
                state.payload.imaging_ready = True
                state.payload.mode = PayloadMode.READY

        if state.payload.imaging_cooldown_s > 0.0:
            state.payload.imaging_cooldown_s = max(
                0.0,
                state.payload.imaging_cooldown_s - dt_s,
            )

    def _start_or_continue_warmup(self, state: SpacecraftState) -> None:
        if state.payload.mode in (PayloadMode.READY, PayloadMode.IMAGING, PayloadMode.PROCESSING):
            state.payload.payload_enabled = True
            state.payload.imaging_ready = True
            return

        if state.eps.battery_soc_pct < self.config.min_battery_soc_for_warmup_pct:
            # Not enough power budget to start warmup
            state.payload.mode = PayloadMode.STANDBY
            state.payload.payload_enabled = False
            state.payload.imaging_ready = False
            return

        state.payload.mode = PayloadMode.WARMUP
        state.payload.payload_enabled = True

        if state.payload.warmup_time_remaining_s <= 0.0:
            state.payload.warmup_time_remaining_s = self.config.warmup_duration_s
            state.payload.imaging_ready = False

    def _maintain_mode(self, state: SpacecraftState) -> None:
        """
        Keep payload mode coherent when no direct command is acting on it.
        """
        if state.payload.mode == PayloadMode.WARMUP:
            if state.payload.warmup_time_remaining_s <= 0.0:
                state.payload.mode = PayloadMode.READY
                state.payload.payload_enabled = True
                state.payload.imaging_ready = True
            return

        if state.payload.mode == PayloadMode.IMAGING:
            # Imaging is transient; after capture, return to READY if allowed.
            if state.payload.imaging_cooldown_s > 0.0:
                state.payload.mode = PayloadMode.READY if state.payload.imaging_ready else PayloadMode.STANDBY
            return

        if state.payload.mode == PayloadMode.PROCESSING:
            # Synthetic "processing" is short-lived; settle back to READY.
            if state.payload.payload_enabled:
                state.payload.mode = PayloadMode.READY if state.payload.imaging_ready else PayloadMode.STANDBY
            return

        if state.payload.payload_enabled and state.payload.imaging_ready:
            if state.payload.mode not in (PayloadMode.READY, PayloadMode.IMAGING, PayloadMode.PROCESSING):
                state.payload.mode = PayloadMode.READY
        elif state.payload.payload_enabled:
            if state.payload.mode not in (PayloadMode.WARMUP,):
                state.payload.mode = PayloadMode.STANDBY
        else:
            if state.payload.mode not in (PayloadMode.OFF, PayloadMode.ERROR):
                state.payload.mode = PayloadMode.STANDBY

    def _reset_payload(self, state: SpacecraftState) -> None:
        state.payload.mode = PayloadMode.OFF
        state.payload.payload_enabled = False
        state.payload.imaging_ready = False
        state.payload.warmup_time_remaining_s = 0.0
        state.payload.imaging_cooldown_s = 0.0
        state.payload.clear_current_frame()

    # -----------------------------------------------------
    # Capture logic
    # -----------------------------------------------------

    def can_capture(self, state: SpacecraftState) -> tuple[bool, str]:
        """
        Validate whether capture is currently operationally allowed.
        """
        if state.payload.has_frame:
            return False, "payload already holds a frame"

        if not state.payload.payload_enabled:
            return False, "payload not enabled"

        if not state.payload.imaging_ready:
            return False, "payload not imaging-ready"

        if state.payload.warmup_time_remaining_s > 0.0:
            return False, "payload still warming up"

        if state.payload.imaging_cooldown_s > 0.0:
            return False, "payload capture cooldown active"

        if state.eps.battery_soc_pct < self.config.min_battery_soc_for_capture_pct:
            return False, "battery too low for capture"

        if not (
            self.config.min_payload_temp_for_capture_c
            <= state.thermal.payload_temp_c
            <= self.config.max_payload_temp_for_capture_c
        ):
            return False, "payload temperature out of capture range"

        if self.config.require_daylight_for_capture and not state.orbit.day_imaging_valid:
            return False, "daylight imaging condition not satisfied"

        if self.config.require_target_for_capture and not state.orbit.over_target:
            return False, "no target available"

        if state.orbit.target_opportunity == TargetOpportunity.NONE:
            return False, "target opportunity is none"

        if self.config.require_good_pointing:
            if state.adcs.pointing_quality < self.config.min_pointing_quality:
                return False, "pointing quality insufficient"

        if self.config.require_settled_attitude and state.adcs.slew_time_remaining_s > 0.0:
            return False, "attitude not settled"

        return True, "capture allowed"

    def _attempt_capture(self, state: SpacecraftState) -> bool:
        can_capture, _ = self.can_capture(state)
        if not can_capture:
            state.payload.mode = PayloadMode.READY if state.payload.imaging_ready else PayloadMode.STANDBY
            return False

        self._frame_counter += 1
        state.payload.mode = PayloadMode.IMAGING

        frame_size_mb = self._generate_frame_size_mb()
        frame_class, cloud_prob, usefulness, confidence = self._generate_synthetic_scene(state)

        state.payload.current_frame_id = self._frame_counter
        state.payload.current_frame_size_mb = frame_size_mb
        state.payload.current_frame_class = frame_class
        state.payload.current_frame_cloud_prob = cloud_prob
        state.payload.current_frame_usefulness = usefulness
        state.payload.classifier_confidence = confidence
        state.payload.classifier_last_latency_s = 0.0
        state.payload.classifier_success = True

        state.payload.total_frames_captured += 1
        if frame_class in (FrameClass.CLEAR, FrameClass.HIGH_VALUE_CLEAR):
            state.payload.total_frames_useful += 1

        state.payload.imaging_cooldown_s = self.config.capture_cooldown_s
        state.payload.mode = PayloadMode.READY

        return True

    # -----------------------------------------------------
    # Synthetic frame generation
    # -----------------------------------------------------

    def _generate_frame_size_mb(self) -> float:
        size = self.config.raw_frame_size_mb_mean + self.rng.uniform(
            -self.config.raw_frame_size_mb_jitter,
            self.config.raw_frame_size_mb_jitter,
        )
        return clamp(
            size,
            self.config.min_frame_size_mb,
            self.config.max_frame_size_mb,
        )

    def _generate_synthetic_scene(
        self,
        state: SpacecraftState,
    ) -> tuple[FrameClass, float, float, float]:
        """
        Generate:
        - frame class
        - cloud probability
        - usefulness
        - confidence-like score

        This is the synthetic stand-in for the classifier+scene pipeline.
        """
        # Start with baseline probabilities
        p_cloudy = self.config.base_cloudy_prob
        p_partly = self.config.base_partly_cloudy_prob
        p_clear = self.config.base_clear_prob
        p_high_value = self.config.base_high_value_clear_prob

        # Target opportunity effects
        if state.orbit.target_opportunity == TargetOpportunity.HIGH_VALUE:
            p_high_value += self.config.high_value_bonus_if_high_value_target
            p_clear += 0.06
            p_cloudy -= 0.08
            p_partly -= 0.04

        elif state.orbit.target_opportunity == TargetOpportunity.VALID:
            p_clear += self.config.clear_bonus_if_valid_target
            p_cloudy -= 0.05
            p_partly -= 0.05

        elif state.orbit.target_opportunity == TargetOpportunity.POOR_LIGHT:
            p_cloudy += self.config.cloudy_bonus_if_poor_light
            p_partly += 0.07
            p_clear -= 0.12
            p_high_value -= 0.05

        # Illumination effects
        sun_elev = state.orbit.sun_elevation_deg
        if sun_elev < 20.0:
            p_cloudy += 0.08
            p_partly += 0.05
            p_clear -= 0.08
            p_high_value -= 0.05
        elif sun_elev > 45.0:
            p_clear += 0.05
            p_high_value += 0.05
            p_cloudy -= 0.04
            p_partly -= 0.06

        # Pointing quality effects
        if state.adcs.pointing_quality == PointingQuality.COARSE:
            p_cloudy += self.config.coarse_pointing_cloud_penalty
            p_clear -= 0.07
            p_high_value -= 0.03
        elif state.adcs.pointing_quality == PointingQuality.PRECISE:
            p_high_value += 0.05
            p_clear += 0.03
            p_cloudy -= 0.04

        # Clamp non-negatives, then renormalize
        probs = [
            max(0.0, p_cloudy),
            max(0.0, p_partly),
            max(0.0, p_clear),
            max(0.0, p_high_value),
        ]
        total = sum(probs)
        if total <= 0.0:
            probs = [1.0, 0.0, 0.0, 0.0]
            total = 1.0
        probs = [p / total for p in probs]

        draw = self.rng.random()
        cumulative = 0.0
        choices = [
            FrameClass.CLOUDY,
            FrameClass.PARTLY_CLOUDY,
            FrameClass.CLEAR,
            FrameClass.HIGH_VALUE_CLEAR,
        ]

        frame_class = FrameClass.CLOUDY
        for cls, p in zip(choices, probs):
            cumulative += p
            if draw <= cumulative:
                frame_class = cls
                break

        # Synthetic cloud probability
        if frame_class == FrameClass.CLOUDY:
            cloud_prob = self.rng.uniform(0.75, 0.98)
        elif frame_class == FrameClass.PARTLY_CLOUDY:
            cloud_prob = self.rng.uniform(0.40, 0.70)
        elif frame_class == FrameClass.CLEAR:
            cloud_prob = self.rng.uniform(0.05, 0.25)
        else:
            cloud_prob = self.rng.uniform(0.01, 0.15)

        # Synthetic usefulness
        usefulness = self._derive_usefulness(
            frame_class=frame_class,
            target_opportunity=state.orbit.target_opportunity,
            sun_elevation_deg=sun_elev,
            pointing_quality=state.adcs.pointing_quality,
        )

        # Confidence-like score
        confidence = self._derive_confidence(
            frame_class=frame_class,
            target_opportunity=state.orbit.target_opportunity,
            pointing_quality=state.adcs.pointing_quality,
            cloud_prob=cloud_prob,
            usefulness=usefulness,
        )

        return frame_class, cloud_prob, usefulness, confidence

    def _derive_usefulness(
        self,
        frame_class: FrameClass,
        target_opportunity: TargetOpportunity,
        sun_elevation_deg: float,
        pointing_quality: PointingQuality,
    ) -> float:
        if frame_class == FrameClass.CLOUDY:
            usefulness = self.rng.uniform(0.00, 0.20)
        elif frame_class == FrameClass.PARTLY_CLOUDY:
            usefulness = self.rng.uniform(0.20, 0.50)
        elif frame_class == FrameClass.CLEAR:
            usefulness = self.rng.uniform(0.55, 0.82)
        else:
            usefulness = self.rng.uniform(0.78, 0.98)

        # Opportunity influence
        if target_opportunity == TargetOpportunity.HIGH_VALUE:
            usefulness += 0.10
        elif target_opportunity == TargetOpportunity.POOR_LIGHT:
            usefulness -= self.config.poor_light_penalty_on_usefulness
        elif target_opportunity == TargetOpportunity.VALID:
            usefulness += 0.03

        # Illumination
        if sun_elevation_deg > 50.0:
            usefulness += self.config.excellent_light_bonus_on_usefulness
        elif sun_elevation_deg < 18.0:
            usefulness -= 0.08

        # Pointing
        if pointing_quality == PointingQuality.COARSE:
            usefulness -= self.config.coarse_pointing_usefulness_penalty
        elif pointing_quality == PointingQuality.PRECISE:
            usefulness += 0.05

        return clamp(
            usefulness,
            self.config.min_usefulness,
            self.config.max_usefulness,
        )

    def _derive_confidence(
        self,
        frame_class: FrameClass,
        target_opportunity: TargetOpportunity,
        pointing_quality: PointingQuality,
        cloud_prob: float,
        usefulness: float,
    ) -> float:
        confidence = self.rng.uniform(self.config.min_confidence, self.config.max_confidence)

        if frame_class == FrameClass.PARTLY_CLOUDY:
            confidence -= 0.10
        elif frame_class == FrameClass.HIGH_VALUE_CLEAR:
            confidence += 0.03

        if target_opportunity == TargetOpportunity.HIGH_VALUE:
            confidence += 0.02
        elif target_opportunity == TargetOpportunity.POOR_LIGHT:
            confidence -= 0.05

        if pointing_quality == PointingQuality.COARSE:
            confidence -= 0.08
        elif pointing_quality == PointingQuality.PRECISE:
            confidence += 0.03

        # Ambiguous cases slightly reduce confidence
        if 0.35 <= cloud_prob <= 0.65:
            confidence -= 0.05

        # High usefulness, clear frame tends to carry stronger confidence
        if usefulness > 0.80 and frame_class in (FrameClass.CLEAR, FrameClass.HIGH_VALUE_CLEAR):
            confidence += 0.03

        return clamp(confidence, self.config.min_confidence, self.config.max_confidence)

    def _refresh_synthetic_inference_metadata(
        self,
        state: SpacecraftState,
        real_frame=None,          # ADD only this parameter
    ) -> None:
        if not state.payload.has_frame:
            state.payload.mode = PayloadMode.ERROR
            state.payload.classifier_success = False
            return

        state.payload.mode = PayloadMode.PROCESSING

        # ── Real classifier path ──────────────────────────────
        if real_frame is not None:
            try:
                from classifier.preprocess   import prepare_input
                from classifier.infer_tflite import CloudClassifier
                from classifier.postprocess  import postprocess

                if not hasattr(self, '_classifier'):
                    self._classifier = CloudClassifier()

                inp        = prepare_input(real_frame)        # (1, 96, 96, 3)
                prediction = self._classifier.predict(inp)
                result     = postprocess(prediction)

                state.payload.current_frame_class         = result['frame_class']
                state.payload.classifier_confidence       = clamp(
                    result['classifier_confidence'],
                    self.config.min_confidence,
                    self.config.max_confidence,
                )
                state.payload.current_frame_cloud_prob    = clamp(
                    result['current_frame_cloud_prob'], 0.0, 1.0
                )
                state.payload.current_frame_usefulness    = clamp(
                    result['current_frame_usefulness'], 0.0, 1.0
                )
                state.payload.classifier_last_latency_s   = result.get(
                    'classifier_last_latency_s', 0.0
                )
                state.payload.classifier_success          = result['classifier_success']
                state.payload.mode = PayloadMode.READY
                return                                    # ← skip synthetic block
            except Exception:
                pass                                      # classifier unavailable → fall through

        # ── Synthetic fallback (your existing code, zero changes) ──
        state.payload.classifier_success = True
        state.payload.classifier_last_latency_s = self.rng.uniform(0.02, 0.15)

        jitter = self.rng.uniform(-0.03, 0.03)
        state.payload.classifier_confidence = clamp(
            state.payload.classifier_confidence + jitter,
            self.config.min_confidence,
            self.config.max_confidence,
        )

        usefulness_jitter = self.rng.uniform(-0.02, 0.02)
        state.payload.current_frame_usefulness = clamp(
            state.payload.current_frame_usefulness + usefulness_jitter,
            self.config.min_usefulness,
            self.config.max_usefulness,
        )

        state.payload.mode = PayloadMode.READY


# =========================================================
# Functional helpers
# =========================================================

def initialize_payload_state(
    state: SpacecraftState,
    config: PayloadConfig = DEFAULT_PAYLOAD_CONFIG,
) -> None:
    subsystem = PayloadSubsystem(config=config)
    subsystem.initialize(state)


def update_payload_state(
    state: SpacecraftState,
    dt_s: float,
    action: Optional[Action] = None,
    action_profile: Optional[ActionEffectProfile] = None,
    config: PayloadConfig = DEFAULT_PAYLOAD_CONFIG,
) -> PayloadBreakdown:
    subsystem = PayloadSubsystem(config=config)
    return subsystem.step(
        state=state,
        dt_s=dt_s,
        action=action,
        action_profile=action_profile,
    )


# =========================================================
# Smoke test helper
# =========================================================

def payload_smoke_summary(
    state: SpacecraftState,
    config: PayloadConfig = DEFAULT_PAYLOAD_CONFIG,
) -> Dict[str, float]:
    """
    Quick payload sanity check helper.
    """
    subsystem = PayloadSubsystem(config=config)
    subsystem.initialize(state)

    # Prepare a valid capture situation
    state.eps.battery_soc_pct = max(state.eps.battery_soc_pct, 60.0)
    state.orbit.over_target = True
    state.orbit.target_opportunity = TargetOpportunity.VALID
    state.orbit.sun_elevation_deg = 45.0
    state.adcs.pointing_error_deg = 0.6
    state.adcs.slew_time_remaining_s = 0.0
    state.payload.payload_enabled = True
    state.payload.imaging_ready = True
    state.payload.mode = PayloadMode.READY

    result = subsystem.step(state=state, dt_s=5.0, action=Action.CAPTURE_IMAGE)

    return {
        "capture_succeeded": 1.0 if result.capture_succeeded else 0.0,
        "frame_generated": 1.0 if result.frame_generated else 0.0,
        "current_frame_id": float(result.current_frame_id),
        "frame_size_mb": result.current_frame_size_mb,
        "cloud_prob": result.current_frame_cloud_prob,
        "usefulness": result.current_frame_usefulness,
        "confidence": result.classifier_confidence,
    }
