from dataclasses import dataclass


@dataclass(frozen=True)
class MissionConfig:
    dt_s: float = 5.0
    max_episode_steps: int = 1080
    max_episode_orbits: int | None = None
