"""
RPM Estimation from IMU Vibration Data

This module provides tools for extracting engine RPM from accelerometer
vibration signatures using spectral analysis techniques.
"""

__version__ = "0.1.0"
__author__ = "Hovercraft Analysis Pipeline"

# Import key components for easier access
from .tracking import RPMFrame
from .cli import main as cli_main

__all__ = ['RPMFrame', 'cli_main']