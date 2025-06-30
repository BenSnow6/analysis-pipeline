"""
Hovercraft Analysis Pipeline

A comprehensive analysis pipeline for processing hovercraft sensor data including
IMU, GPS, and other telemetry data.
"""

__version__ = "1.0.0"
__author__ = "Hovercraft Analysis Team"

# Import key components for easy access
from . import core
from . import analysis
from . import apps
from . import scripts

# Expose commonly used functions at package level
from .core import (
    get_experiment_path,
    get_all_experiment_names,
    load_csv_data,
    save_csv_data,
    load_hdf5_data,
    save_hdf5_data,
    config,
)

__all__ = [
    # Modules
    "core",
    "analysis", 
    "apps",
    "scripts",
    
    # Common functions
    "get_experiment_path",
    "get_all_experiment_names",
    "load_csv_data",
    "save_csv_data",
    "load_hdf5_data",
    "save_hdf5_data",
    "config",
    
    # Package info
    "__version__",
    "__author__",
]