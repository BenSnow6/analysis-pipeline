"""
Central path configuration for the hovercraft analysis pipeline.
This module provides consistent path definitions across the entire codebase.
"""

from pathlib import Path
from typing import List, Optional

# Base directories - updated for new package structure
_PACKAGE_DIR = Path(__file__).parent.parent.absolute()  # src dir
PROJECT_ROOT = _PACKAGE_DIR.parent.absolute()  # Repository root (analysis-pipeline)
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"
NOTES_DIR = PROJECT_ROOT / "notes"

# Core package directories
CORE_DIR = _PACKAGE_DIR / "core"
ANALYSIS_DIR = _PACKAGE_DIR / "analysis"
APPS_DIR = _PACKAGE_DIR / "apps"
SCRIPTS_DIR = _PACKAGE_DIR / "scripts"

# Analysis module directories
ALIGNMENT_DIR = ANALYSIS_DIR / "alignment"
ORIENTATION_DIR = ANALYSIS_DIR / "orientation"
TIMESTAMP_DIR = ANALYSIS_DIR / "timestamp"
RPM_DIR = ANALYSIS_DIR / "rpm"

# App directories
DASHBOARD_DIR = APPS_DIR / "dashboard"

# Raw data paths (new structure as of Phase 4)
RAW_DATA_DIR = DATA_DIR / "raw"
MORNING_DATA_DIR = RAW_DATA_DIR / "morning"
AFTERNOON_DATA_DIR = RAW_DATA_DIR / "afternoon"
MORNING_EXPERIMENTS_DIR = MORNING_DATA_DIR / "Experiments"
AFTERNOON_EXPERIMENTS_DIR = AFTERNOON_DATA_DIR / "Experiments"

# Processed data paths
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ALIGNED_DATA_DIR = PROCESSED_DATA_DIR / "aligned"
ORIENTATION_DATA_DIR = PROCESSED_DATA_DIR / "orientation"
RPM_DATA_DIR = PROCESSED_DATA_DIR / "rpm"
TIMESTAMP_DATA_DIR = PROCESSED_DATA_DIR / "timestamp"

# Cache directory
CACHE_DIR = DATA_DIR / "cache"

# Results and documentation
RESULTS_DIR = DOCS_DIR / "results"
TIMESTAMP_RESULTS_DIR = TIMESTAMP_DATA_DIR / "timestamp_analysis_results"
CODEBASE_ANALYSIS_DIR = DOCS_DIR / "codebase_analysis"

# Configuration files
CONFIG_DIR = PROJECT_ROOT / "config"  # Centralized config directory
EXPERIMENT_MAPPING_FILE = CONFIG_DIR / "experiments" / "experiment_mapping.json"
ORIENTATION_CONFIG_FILE = CONFIG_DIR / "processing" / "orientation_config.yaml"
SENSOR_ORIENTATIONS_FILE = CONFIG_DIR / "sensors" / "sensor_orientations.json"
EXPERIMENT_MANIFEST_FILE = CONFIG_DIR / "experiments" / "experiment_manifest.yaml"


def get_experiment_path(experiment_name: str, time_of_day: str = "morning") -> Path:
    """
    Get the path to a specific experiment.

    Args:
        experiment_name: Name of the experiment (e.g., "007_Fast_stbd_turn_1")
        time_of_day: Either "morning" or "afternoon"

    Returns:
        Path to the experiment directory

    Raises:
        ValueError: If time_of_day is not "morning" or "afternoon"
    """
    if time_of_day == "morning":
        return MORNING_EXPERIMENTS_DIR / experiment_name
    elif time_of_day == "afternoon":
        return AFTERNOON_EXPERIMENTS_DIR / experiment_name
    else:
        raise ValueError(
            f"time_of_day must be 'morning' or 'afternoon', got '{time_of_day}'"
        )


def get_all_experiment_names(time_of_day: Optional[str] = None) -> List[str]:
    """
    Get all experiment names.

    Args:
        time_of_day: Optional filter for "morning" or "afternoon" experiments

    Returns:
        List of experiment directory names

    Raises:
        ValueError: If time_of_day is not None, "morning", or "afternoon"
    """
    experiments = []

    if time_of_day is None:
        times = ["morning", "afternoon"]
    elif time_of_day in ["morning", "afternoon"]:
        times = [time_of_day]
    else:
        raise ValueError(
            f"time_of_day must be None, 'morning', or 'afternoon', got '{time_of_day}'"
        )

    for time in times:
        exp_dir = (
            MORNING_EXPERIMENTS_DIR if time == "morning" else AFTERNOON_EXPERIMENTS_DIR
        )
        if exp_dir.exists():
            experiments.extend([d.name for d in exp_dir.iterdir() if d.is_dir()])

    return sorted(list(set(experiments)))  # Remove duplicates and sort


def get_aligned_data_path(experiment_name: str, time_of_day: str = "morning") -> Path:
    """
    Get the path to aligned data for a specific experiment.

    Args:
        experiment_name: Name of the experiment
        time_of_day: Either "morning" or "afternoon"

    Returns:
        Path to the aligned data directory
    """
    return ALIGNED_DATA_DIR / time_of_day / experiment_name


def ensure_directories():
    """Create essential directories if they don't exist."""
    essential_dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        DOCS_DIR,
        RESULTS_DIR,
        PROCESSED_DATA_DIR,
        ALIGNED_DATA_DIR,
        ALIGNED_DATA_DIR / "morning",
        ALIGNED_DATA_DIR / "afternoon",
        ORIENTATION_DATA_DIR,
        RPM_DATA_DIR,
        TIMESTAMP_DATA_DIR,
        CACHE_DIR,
    ]

    for directory in essential_dirs:
        directory.mkdir(parents=True, exist_ok=True)


# Backward compatibility - will be removed in future versions
def get_legacy_aligned_path(experiment_name: str, time_of_day: str = "morning") -> Path:
    """
    Get path to legacy aligned data location.
    This is for backward compatibility during migration.
    """
    # Old location was in code/processed/aligned/
    legacy_base = PROJECT_ROOT / "code" / "alignment_analysis" / "aligned_data"
    if time_of_day == "afternoon":
        return legacy_base / "afternoon" / f"{experiment_name}_aligned.h5"
    else:
        return legacy_base / f"{experiment_name}_aligned.h5"


if __name__ == "__main__":
    # When run directly, print path information
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Package directory: {_PACKAGE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Docs directory: {DOCS_DIR}")
    print(f"\nConfiguration files:")
    print(f"  Experiment mapping: {EXPERIMENT_MAPPING_FILE}")
    print(f"  Sensor orientations: {SENSOR_ORIENTATIONS_FILE}")
    print(f"\nExperiment directories:")
    print(f"  Morning: {MORNING_EXPERIMENTS_DIR}")
    print(f"  Afternoon: {AFTERNOON_EXPERIMENTS_DIR}")
    print(f"\nProcessed data directories:")
    print(f"  Aligned: {ALIGNED_DATA_DIR}")
    print(f"  Orientation: {ORIENTATION_DATA_DIR}")
    print(f"  RPM: {RPM_DATA_DIR}")
    print(f"  Timestamp: {TIMESTAMP_DATA_DIR}")
