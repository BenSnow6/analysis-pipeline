"""
Data I/O operations for RPM estimation.

This module handles loading sensor data, configuration files,
and saving results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Loaded configuration from {config_path}")
    return config


def save_config(config: dict, config_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.debug(f"Saved configuration to {config_path}")


def find_experiment_data(experiment: str, session: str, base_path: Optional[Path] = None) -> Path:
    """
    Find the data directory for a given experiment.
    
    Args:
        experiment: Experiment name (e.g., '026_Engine_rpm_sweep')
        session: 'morning' or 'afternoon'
        base_path: Base directory for data (defaults to relative path)
        
    Returns:
        Path to experiment data directory
        
    Raises:
        FileNotFoundError: If experiment data not found
    """
    if base_path is None:
        # Default to relative path from this module
        module_path = Path(__file__).parent
        base_path = module_path.parent.parent / 'hovercraft_data_analysis' / 'alignment_analysis' / 'aligned_data'
    
    # Check for CSV export directory first
    csv_dir = base_path / f"{experiment}_csv"
    if csv_dir.exists():
        logger.debug(f"Found CSV data directory: {csv_dir}")
        return csv_dir
    
    # Check session-specific directory
    session_dir = base_path / session / f"{experiment}_csv"
    if session_dir.exists():
        logger.debug(f"Found session-specific CSV directory: {session_dir}")
        return session_dir
    
    # Check for HDF5 file as fallback
    hdf5_file = base_path / session / f"{experiment}_aligned.h5"
    if hdf5_file.exists():
        logger.warning(f"Found HDF5 file but no CSV export: {hdf5_file}")
        logger.warning("Run export_to_csv.py first to convert HDF5 to CSV format")
    
    raise FileNotFoundError(
        f"No data found for experiment {experiment} ({session} session) in {base_path}"
    )


def load_sensor_data(experiment: str, session: str, sensor_id: str, 
                    base_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load sensor data for a specific experiment and sensor.
    
    Args:
        experiment: Experiment name
        session: 'morning' or 'afternoon'
        sensor_id: Sensor identifier (e.g., 'Sensor_3')
        base_path: Base directory for data
        
    Returns:
        DataFrame with sensor data (time, x, y, z, gyro_x, gyro_y, gyro_z)
    """
    data_dir = find_experiment_data(experiment, session, base_path)
    sensor_file = data_dir / f"{sensor_id}.csv"
    
    if not sensor_file.exists():
        raise FileNotFoundError(f"Sensor data not found: {sensor_file}")
    
    # Load CSV data
    df = pd.read_csv(sensor_file)
    
    # Validate expected columns
    expected_cols = ['t', 'x', 'y', 'z', 'time_from_sync', 'gyro_x', 'gyro_y', 'gyro_z']
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}")
    
    logger.info(f"Loaded {len(df)} samples from {sensor_id}")
    return df


def load_gps_data(experiment: str, session: str, base_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load GPS data for a specific experiment.
    
    Args:
        experiment: Experiment name
        session: 'morning' or 'afternoon'
        base_path: Base directory for data
        
    Returns:
        DataFrame with GPS data
    """
    data_dir = find_experiment_data(experiment, session, base_path)
    gps_file = data_dir / "gps.csv"
    
    if gps_file.exists():
        return pd.read_csv(gps_file)
    else:
        logger.warning(f"GPS data not found: {gps_file}")
        return pd.DataFrame()


def save_rpm_results(rpm_data: pd.DataFrame, output_path: Path) -> None:
    """
    Save RPM estimation results to CSV.
    
    Args:
        rpm_data: DataFrame with RPM results
        output_path: Path to save results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rpm_data.to_csv(output_path, index=False)
    logger.info(f"Saved RPM results to {output_path}")


def list_available_experiments(base_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    List all available experiments by session.
    
    Returns:
        Dictionary mapping session to list of experiment names
    """
    if base_path is None:
        module_path = Path(__file__).parent
        base_path = module_path.parent.parent / 'hovercraft_data_analysis' / 'alignment_analysis' / 'aligned_data'
    
    experiments = {'morning': [], 'afternoon': []}
    
    # Check root directory for CSV exports
    for csv_dir in base_path.glob("*_csv"):
        exp_name = csv_dir.name.replace("_csv", "")
        # Try to determine session from content or default to afternoon
        experiments['afternoon'].append(exp_name)
    
    # Check session-specific directories
    for session in ['morning', 'afternoon']:
        session_dir = base_path / session
        if session_dir.exists():
            for csv_dir in session_dir.glob("*_csv"):
                exp_name = csv_dir.name.replace("_csv", "")
                if exp_name not in experiments[session]:
                    experiments[session].append(exp_name)
    
    return experiments