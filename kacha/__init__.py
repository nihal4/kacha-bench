"""KachaBench: rural monsoon-road terrain for legged locomotion RL."""

from .terrain import RoadConfig, generate_village_road
from .scene import build_model

__version__ = "0.0.1"
__all__ = ["RoadConfig", "generate_village_road", "build_model"]
