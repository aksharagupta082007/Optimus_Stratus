from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from models.action_space import ActionEffectProfile
from models.enums import Action, DataHandlingMode, FrameClass, ResetCause
from models.state_models import SpacecraftState


# =========================================================
# Configuration
# =========================================================

@dataclass(frozen=True)
class CDHConfig:
    """
    Command and Data Handling (CDH) / onboard data system model.

    This is an operational data-management simulator for a CubeSat mission.
    It focuses on what matters for RL and mission scheduling:

    - raw frame buffering
    - classification-aware storage decisions
    - compression
    - useful-data queue growth
    - memory pressure
    - purge / discard handling
    - reset / recovery effects
    - storage corruption and filesystem health guards

    Units:
    - memory, buffers, queues: MB
    - time: s
    """

    # -----------------------------------------------------
    # Storage capacity
    # -----------------------------------------------------
    memory_capacity_mb: float = 4096.0
    safety_reserve_mb: float = 128.0

    # -----------------------------------------------------
    # Frame size assumptions
    # -----------------------------------------------------
    default_raw_frame_mb: float = 32.0
    default_clear_stored_mb: float = 16.0
    default_high_value_stored_mb: float = 20.0
    default_cloudy_stored_mb: float = 12.0

    # -----------------------------------------------------
    # Compression / representation
    # -----------------------------------------------------
    compression_ratio_default: float = 0.50
    high_value_compression_ratio: float = 0.60
    cloudy_compression_ratio: float = 0.40

    # -----------------------------------------------------
    # Background / processing behavior
    # -----------------------------------------------------
    buffering_cpu_load_pct: float = 18.0
    classifying_cpu_load_pct: float = 42.0
    compressing_cpu_load_pct: float = 55.0
    downlinking_cpu_load_pct: float = 30.0
    recovery_cpu_load_pct: float = 38.0
    idle_cpu_load_pct: float = 10.0

    # -----------------------------------------------------
    # Storage reliability policy
    # -----------------------------------------------------
    critical_memory_utilization_frac: float = 0.95
    overflow_memory_utilization_frac: float = 1.02

    purge_fraction_on_recovery: float = 0.10
    reset_clears_raw_buffer: bool = True
    reset_keeps_downlink_queue: bool = True

    # -----------------------------------------------------
    # Numerical guards
    # -----------------------------------------------------
    min_memory_mb: float = 0.0
    max_memory_mb: float = 100_000.0


DEFAULT_CDH_CONFIG = CDHConfig()


# =========================================================
# Helpers
# =========================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# =========================================================
# Breakdown
# =========================================================

@dataclass(frozen=True)
class CDHBreakdown:
    """
    Detailed CDH update report for debugging, testing, and reward interpretation.
    """
    mode: str
    memory_used_before_mb: float
    memory_used_after_mb: float
    memory_free_after_mb: float
    raw_buffer_mb: float
    processed_buffer_mb: float
    downlink_queue_mb: float
    cpu_load_pct: float
    stored_this_step_mb: float
    purged_this_step_mb: float
    discarded_this_step_mb: float
    compression_applied: bool
    filesystem_healthy: bool
    storage_corrupted: bool


# =========================================================
# CDH subsystem
# =========================================================

class CDHSubsystem:
    """
    Command and Data Handling subsystem.

    Responsibilities:
    - manage onboard storage and queue accounting
    - store and discard frames
    - compress frame products
    - maintain CPU load / data handling mode
    - apply purge / recovery logic
    - detect memory pressure and corruption conditions
    """

    def __init__(self, config: CDHConfig = DEFAULT_CDH_CONFIG):
        self.config = config

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def initialize(self, state: SpacecraftState) -> None:
        state.cdh.memory_capacity_mb = self.config.memory_capacity_mb
        state.cdh.memory_used_mb = clamp(
            state.cdh.memory_used_mb,
            self.config.min_memory_mb,
            self.config.memory_capacity_mb,
        )
        state.cdh.raw_buffer_mb = clamp(state.cdh.raw_buffer_mb, 0.0, self.config.memory_capacity_mb)
        state.cdh.processed_buffer_mb = clamp(state.cdh.processed_buffer_mb, 0.0, self.config.memory_capacity_mb)
        state.cdh.downlink_queue_mb = clamp(state.cdh.downlink_queue_mb, 0.0, self.config.memory_capacity_mb)
        state.cdh.compression_ratio = self.config.compression_ratio_default
        state.cdh.cpu_load_pct = self.config.idle_cpu_load_pct
        state.cdh.filesystem_healthy = True if state.cdh.filesystem_healthy is None else state.cdh.filesystem_healthy
        state.cdh.storage_corrupted = False if state.cdh.storage_corrupted is None else state.cdh.storage_corrupted
        self._reconcile_memory(state)

    def step(
        self,
        state: SpacecraftState,
        dt_s: float,
        action: Optional[Action] = None,
        action_profile: Optional[ActionEffectProfile] = None,
    ) -> CDHBreakdown:
        """
        Advance CDH state by one simulation step.
        """
        memory_before = state.cdh.memory_used_mb
        stored_this_step_mb = 0.0
        purged_this_step_mb = 0.0
        discarded_this_step_mb = 0.0
        compression_applied = False

        self._apply_action_mode_hint(state, action)
        self._update_cpu_load(state)

        if action == Action.STORE_FRAME:
            stored_this_step_mb = self._store_current_frame(state)

        elif action == Action.DISCARD_FRAME:
            discarded_this_step_mb = self._discard_current_frame(state)

        elif action == Action.COMPRESS_DATA:
            compression_applied = self._compress_current_frame(state)

        elif action == Action.RESET_CDH:
            purged_this_step_mb = self._reset_cdh(state)

        elif action == Action.FAULT_RECOVERY:
            purged_this_step_mb = self._recovery_purge(state)

        # Background consistency pass
        self._reconcile_memory(state)
        self._update_health_flags(state)

        return CDHBreakdown(
            mode=state.cdh.mode.name,
            memory_used_before_mb=memory_before,
            memory_used_after_mb=state.cdh.memory_used_mb,
            memory_free_after_mb=state.cdh.memory_free_mb,
            raw_buffer_mb=state.cdh.raw_buffer_mb,
            processed_buffer_mb=state.cdh.processed_buffer_mb,
            downlink_queue_mb=state.cdh.downlink_queue_mb,
            cpu_load_pct=state.cdh.cpu_load_pct,
            stored_this_step_mb=stored_this_step_mb,
            purged_this_step_mb=purged_this_step_mb,
            discarded_this_step_mb=discarded_this_step_mb,
            compression_applied=compression_applied,
            filesystem_healthy=state.cdh.filesystem_healthy,
            storage_corrupted=state.cdh.storage_corrupted,
        )

    # -----------------------------------------------------
    # Frame ingestion / storage API
    # -----------------------------------------------------

    def ingest_captured_frame(
        self,
        state: SpacecraftState,
        frame_size_mb: Optional[float] = None,
    ) -> bool:
        """
        Called by payload subsystem after a successful capture.
        Adds a raw frame into the transient raw buffer if capacity allows.
        """
        size_mb = frame_size_mb if frame_size_mb is not None else (
            state.payload.current_frame_size_mb or self.config.default_raw_frame_mb
        )
        size_mb = max(0.0, size_mb)

        if not self.can_accept_raw_frame(state, size_mb):
            return False

        state.cdh.raw_buffer_mb += size_mb
        self._reconcile_memory(state)
        return True

    def can_accept_raw_frame(self, state: SpacecraftState, frame_size_mb: float) -> bool:
        projected = state.cdh.memory_used_mb + frame_size_mb
        limit = max(0.0, state.cdh.memory_capacity_mb - self.config.safety_reserve_mb)
        return projected <= limit and state.cdh.filesystem_healthy and not state.cdh.storage_corrupted

    # -----------------------------------------------------
    # Core storage logic
    # -----------------------------------------------------

    def _store_current_frame(self, state: SpacecraftState) -> float:
        """
        Store the currently held frame from payload into managed storage and queue.
        """
        if not state.payload.has_frame:
            return 0.0

        frame_class = state.payload.current_frame_class
        raw_size_mb = max(
            state.payload.current_frame_size_mb,
            self.config.default_raw_frame_mb,
        )

        # If no raw frame exists in CDH buffer yet, synthesize it now so accounting stays consistent.
        if state.cdh.raw_buffer_mb < raw_size_mb:
            additional_needed = raw_size_mb - state.cdh.raw_buffer_mb
            if not self.can_accept_raw_frame(state, additional_needed):
                return 0.0
            state.cdh.raw_buffer_mb += additional_needed

        stored_size_mb = self._compressed_size_for_frame(frame_class, raw_size_mb)

        # Remove raw buffer representation, add processed/queued representation
        state.cdh.raw_buffer_mb = max(0.0, state.cdh.raw_buffer_mb - raw_size_mb)
        state.cdh.processed_buffer_mb += stored_size_mb
        state.cdh.downlink_queue_mb += stored_size_mb
        state.cdh.total_useful_mb_stored += stored_size_mb

        # Clear payload-held frame after successful storage
        state.payload.clear_current_frame()

        self._reconcile_memory(state)
        return stored_size_mb

    def _discard_current_frame(self, state: SpacecraftState) -> float:
        """
        Discard payload-held frame and remove any corresponding raw transient buffer if present.
        """
        if not state.payload.has_frame:
            return 0.0

        raw_size_mb = max(
            state.payload.current_frame_size_mb,
            self.config.default_raw_frame_mb,
        )
        removed_mb = min(state.cdh.raw_buffer_mb, raw_size_mb)
        state.cdh.raw_buffer_mb = max(0.0, state.cdh.raw_buffer_mb - raw_size_mb)

        # Accounting in payload statistics
        state.payload.total_frames_discarded += 1
        state.payload.clear_current_frame()

        self._reconcile_memory(state)
        return removed_mb

    def _compress_current_frame(self, state: SpacecraftState) -> bool:
        """
        Compression here means: prepare the current frame for more efficient storage.
        We model this by updating the CDH compression ratio context and processed buffer.
        """
        if not state.payload.has_frame:
            return False

        frame_class = state.payload.current_frame_class
        raw_size_mb = max(
            state.payload.current_frame_size_mb,
            self.config.default_raw_frame_mb,
        )
        compressed_size_mb = self._compressed_size_for_frame(frame_class, raw_size_mb)

        # Compression is represented as a staged processed buffer product.
        # Do not duplicate if already present in processed buffer for same frame-like action pattern.
        state.cdh.processed_buffer_mb = max(state.cdh.processed_buffer_mb, compressed_size_mb)
        state.cdh.compression_ratio = compressed_size_mb / max(raw_size_mb, 1e-9)

        self._reconcile_memory(state)
        return True

    # -----------------------------------------------------
    # Recovery / reset
    # -----------------------------------------------------

    def _recovery_purge(self, state: SpacecraftState) -> float:
        """
        Purge a fraction of less critical processed data to recover headroom.
        Queue is reduced proportionally.
        """
        purge_candidate_mb = state.cdh.processed_buffer_mb * self.config.purge_fraction_on_recovery
        purge_candidate_mb = max(0.0, purge_candidate_mb)

        if purge_candidate_mb <= 0.0:
            return 0.0

        queue_purge = min(state.cdh.downlink_queue_mb, purge_candidate_mb)
        processed_purge = min(state.cdh.processed_buffer_mb, purge_candidate_mb)

        state.cdh.processed_buffer_mb -= processed_purge
        state.cdh.downlink_queue_mb -= queue_purge

        self._reconcile_memory(state)
        return processed_purge

    def _reset_cdh(self, state: SpacecraftState) -> float:
        """
        Controlled CDH reset:
        - clears transient buffers
        - may keep downlink queue depending on config
        - resets CPU load and mode
        """
        purged = 0.0

        if self.config.reset_clears_raw_buffer:
            purged += state.cdh.raw_buffer_mb
            state.cdh.raw_buffer_mb = 0.0

        if not self.config.reset_keeps_downlink_queue:
            purged += state.cdh.downlink_queue_mb
            state.cdh.processed_buffer_mb = max(0.0, state.cdh.processed_buffer_mb - state.cdh.downlink_queue_mb)
            state.cdh.downlink_queue_mb = 0.0

        state.cdh.mode = DataHandlingMode.RECOVERY
        state.cdh.cpu_load_pct = self.config.recovery_cpu_load_pct
        state.cdh.last_reset_cause = ResetCause.COMMAND
        state.cdh.reset_count += 1
        state.cdh.filesystem_healthy = True
        state.cdh.storage_corrupted = False

        self._reconcile_memory(state)
        return purged

    # -----------------------------------------------------
    # Mode / CPU load
    # -----------------------------------------------------

    def _apply_action_mode_hint(self, state: SpacecraftState, action: Optional[Action]) -> None:
        if action is None:
            return

        mapping = {
            Action.RUN_CLASSIFIER: DataHandlingMode.CLASSIFYING,
            Action.STORE_FRAME: DataHandlingMode.BUFFERING,
            Action.DISCARD_FRAME: DataHandlingMode.PURGING,
            Action.COMPRESS_DATA: DataHandlingMode.COMPRESSING,
            Action.PREPARE_DOWNLINK: DataHandlingMode.QUEUEING,
            Action.DOWNLINK_LOW_RATE: DataHandlingMode.DOWNLINKING,
            Action.DOWNLINK_HIGH_RATE: DataHandlingMode.DOWNLINKING,
            Action.RESET_CDH: DataHandlingMode.RECOVERY,
            Action.FAULT_RECOVERY: DataHandlingMode.RECOVERY,
        }
        state.cdh.mode = mapping.get(action, state.cdh.mode)

    def _update_cpu_load(self, state: SpacecraftState) -> None:
        mode = state.cdh.mode
        if mode == DataHandlingMode.IDLE:
            state.cdh.cpu_load_pct = self.config.idle_cpu_load_pct
        elif mode == DataHandlingMode.BUFFERING:
            state.cdh.cpu_load_pct = self.config.buffering_cpu_load_pct
        elif mode == DataHandlingMode.CLASSIFYING:
            state.cdh.cpu_load_pct = self.config.classifying_cpu_load_pct
        elif mode == DataHandlingMode.COMPRESSING:
            state.cdh.cpu_load_pct = self.config.compressing_cpu_load_pct
        elif mode == DataHandlingMode.DOWNLINKING:
            state.cdh.cpu_load_pct = self.config.downlinking_cpu_load_pct
        elif mode == DataHandlingMode.RECOVERY:
            state.cdh.cpu_load_pct = self.config.recovery_cpu_load_pct
        else:
            state.cdh.cpu_load_pct = self.config.idle_cpu_load_pct

    # -----------------------------------------------------
    # Memory accounting / health
    # -----------------------------------------------------

    def _compressed_size_for_frame(self, frame_class: FrameClass, raw_size_mb: float) -> float:
        if frame_class == FrameClass.HIGH_VALUE_CLEAR:
            ratio = self.config.high_value_compression_ratio
        elif frame_class == FrameClass.CLOUDY:
            ratio = self.config.cloudy_compression_ratio
        else:
            ratio = self.config.compression_ratio_default

        return max(0.1, raw_size_mb * ratio)

    def _reconcile_memory(self, state: SpacecraftState) -> None:
        """
        Recompute total memory used from components.
        """
        total_used = (
            max(0.0, state.cdh.raw_buffer_mb)
            + max(0.0, state.cdh.processed_buffer_mb)
        )
        state.cdh.memory_used_mb = clamp(
            total_used,
            self.config.min_memory_mb,
            self.config.max_memory_mb,
        )

        # Clamp components to non-negative
        state.cdh.raw_buffer_mb = max(0.0, state.cdh.raw_buffer_mb)
        state.cdh.processed_buffer_mb = max(0.0, state.cdh.processed_buffer_mb)
        state.cdh.downlink_queue_mb = max(0.0, state.cdh.downlink_queue_mb)

        # Queue cannot exceed processed buffer in this operational abstraction
        if state.cdh.downlink_queue_mb > state.cdh.processed_buffer_mb:
            state.cdh.downlink_queue_mb = state.cdh.processed_buffer_mb

    def _update_health_flags(self, state: SpacecraftState) -> None:
        util = state.cdh.memory_used_mb / max(state.cdh.memory_capacity_mb, 1e-9)

        if util >= self.config.overflow_memory_utilization_frac:
            state.cdh.filesystem_healthy = False
            state.cdh.storage_corrupted = True
        elif util >= self.config.critical_memory_utilization_frac:
            # still healthy, but at risk; leave corruption false
            state.cdh.filesystem_healthy = True if not state.cdh.storage_corrupted else False
        else:
            if not state.cdh.storage_corrupted:
                state.cdh.filesystem_healthy = True


# =========================================================
# Functional helpers
# =========================================================

def initialize_cdh_state(
    state: SpacecraftState,
    config: CDHConfig = DEFAULT_CDH_CONFIG,
) -> None:
    subsystem = CDHSubsystem(config=config)
    subsystem.initialize(state)


def update_cdh_state(
    state: SpacecraftState,
    dt_s: float,
    action: Optional[Action] = None,
    action_profile: Optional[ActionEffectProfile] = None,
    config: CDHConfig = DEFAULT_CDH_CONFIG,
) -> CDHBreakdown:
    subsystem = CDHSubsystem(config=config)
    return subsystem.step(
        state=state,
        dt_s=dt_s,
        action=action,
        action_profile=action_profile,
    )


# =========================================================
# Smoke test helper
# =========================================================

def cdh_smoke_summary(
    state: SpacecraftState,
    config: CDHConfig = DEFAULT_CDH_CONFIG,
) -> Dict[str, float]:
    """
    Quick CDH sanity-check helper.
    """
    subsystem = CDHSubsystem(config=config)
    subsystem.initialize(state)

    mem_before = state.cdh.memory_used_mb
    queue_before = state.cdh.downlink_queue_mb

    # Synthetic frame injection
    state.payload.current_frame_id = 1
    state.payload.current_frame_size_mb = config.default_raw_frame_mb
    state.payload.current_frame_class = FrameClass.CLEAR

    ingest_ok = subsystem.ingest_captured_frame(state)
    store_breakdown = subsystem.step(state, dt_s=5.0, action=Action.STORE_FRAME)

    return {
        "ingest_ok": 1.0 if ingest_ok else 0.0,
        "memory_before_mb": mem_before,
        "memory_after_mb": state.cdh.memory_used_mb,
        "queue_before_mb": queue_before,
        "queue_after_mb": state.cdh.downlink_queue_mb,
        "stored_this_step_mb": store_breakdown.stored_this_step_mb,
        "cpu_load_pct": state.cdh.cpu_load_pct,
        "filesystem_healthy": 1.0 if state.cdh.filesystem_healthy else 0.0,
        "storage_corrupted": 1.0 if state.cdh.storage_corrupted else 0.0,
    }
