from dataclasses import dataclass


@dataclass(frozen=True)
class RLConfig:
    observation_type: str = "bundle"
    include_action_mask_in_obs: bool = True

    invalid_action_uses_soft_penalty: bool = True
    invalid_action_fallback_to_noop: bool = True
