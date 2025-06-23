"""
Orientation analysis package for hovercraft sensor data.
Validates sensor rotation matrices and estimates biases.
"""

from .orientation_check import OrientationChecker
from .rotation_validator import RotationValidator
from .static_detector import StaticDetector
from .dynamic_validator import DynamicValidator
from .bias_estimator import BiasEstimator
from .plot_orientation import OrientationPlotter

__version__ = "0.1.0"
__all__ = [
    "OrientationChecker",
    "RotationValidator", 
    "StaticDetector",
    "DynamicValidator",
    "BiasEstimator",
    "OrientationPlotter"
]