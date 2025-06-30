"""
Core utilities for the hovercraft analysis pipeline.
"""

from .config import Config, ConfigManager, config, get_config
from .io import (
    ensure_timestamp_index,
    load_csv_data,
    load_experiment_data,
    load_hdf5_data,
    save_csv_data,
    save_hdf5_data,
    validate_dataframe,
)
from .paths import (  # Base directories; Raw data directories; Processed data; Cache; Configuration files; Helper functions
    AFTERNOON_DATA_DIR,
    AFTERNOON_EXPERIMENTS_DIR,
    ALIGNED_DATA_DIR,
    CACHE_DIR,
    DATA_DIR,
    DOCS_DIR,
    EXPERIMENT_MANIFEST_FILE,
    EXPERIMENT_MAPPING_FILE,
    MORNING_DATA_DIR,
    MORNING_EXPERIMENTS_DIR,
    ORIENTATION_CONFIG_FILE,
    ORIENTATION_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RPM_DATA_DIR,
    SENSOR_ORIENTATIONS_FILE,
    TESTS_DIR,
    TIMESTAMP_DATA_DIR,
    ensure_directories,
    get_aligned_data_path,
    get_all_experiment_names,
    get_experiment_path,
)

__all__ = [
    # Config
    "Config",
    "ConfigManager",
    "config",
    "get_config",
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "DOCS_DIR",
    "TESTS_DIR",
    "RAW_DATA_DIR",
    "MORNING_DATA_DIR",
    "AFTERNOON_DATA_DIR",
    "MORNING_EXPERIMENTS_DIR",
    "AFTERNOON_EXPERIMENTS_DIR",
    "PROCESSED_DATA_DIR",
    "ALIGNED_DATA_DIR",
    "ORIENTATION_DATA_DIR",
    "RPM_DATA_DIR",
    "TIMESTAMP_DATA_DIR",
    "CACHE_DIR",
    "EXPERIMENT_MAPPING_FILE",
    "SENSOR_ORIENTATIONS_FILE",
    "ORIENTATION_CONFIG_FILE",
    "EXPERIMENT_MANIFEST_FILE",
    "get_experiment_path",
    "get_all_experiment_names",
    "get_aligned_data_path",
    "ensure_directories",
    # I/O
    "load_csv_data",
    "save_csv_data",
    "load_hdf5_data",
    "save_hdf5_data",
    "load_experiment_data",
    "validate_dataframe",
    "ensure_timestamp_index",
]
