"""
Data loading utilities for timestamp analysis.

This module handles loading sensor data from CSV files and configuration
from YAML specifications.
"""

import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Constants matching the dashboard app structure
DATA_REPO_PATH = 'data/raw'
GPS_SUBDIR = "GPS"
IMU_SUBDIR = "IMU"

# Sensor directory mapping
SENSOR_DIR_MAP = {
    "sensor_3": "Sensor_3",
    "sensor_4": "Sensor_4",
    "sensor_5": "Sensor_5",
    "sensor_wb": "Sensor_wb",
    "sensor_wnb": "Sensor_wnb",
}

# IMU measurement types
IMU_MEASUREMENT_TYPES = ['accel', 'gyro', 'angle', 'mag']


def load_sensor_specs(spec_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load sensor specifications from YAML file.
    
    Args:
        spec_path: Path to sensor_specs.yaml. If None, uses default location.
        
    Returns:
        Dictionary containing sensor specifications.
    """
    if spec_path is None:
        # Use centralized config location
        spec_path = Path(__file__).parent.parent.parent.parent / 'config' / 'sensors' / 'sensor_specs.yaml'
    
    try:
        with open(spec_path, 'r') as f:
            specs = yaml.safe_load(f)
        return specs
    except FileNotFoundError:
        warnings.warn(f"Sensor spec file not found at {spec_path}. Using defaults.")
        return get_default_specs()
    except yaml.YAMLError as e:
        warnings.warn(f"Error parsing sensor spec file: {e}. Using defaults.")
        return get_default_specs()


def get_default_specs() -> Dict[str, Any]:
    """Return default sensor specifications."""
    return {
        'sensors': {
            'gps': {
                'expected_rate_hz': 1,
                'jitter_threshold_ms': 100,
                'gap_threshold_factor': 2.0
            },
            'default': {
                'expected_rate_hz': 100,
                'jitter_threshold_ms': 20,
                'gap_threshold_factor': 10.0
            }
        },
        'analysis': {
            'min_samples': 100,
            'auto_detect_rate': True,
            'rate_tolerance_percent': 10
        }
    }


def get_sensor_config(sensor_name: str, specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get configuration for a specific sensor.
    
    Args:
        sensor_name: Name of the sensor (e.g., 'gps', 'sensor_3')
        specs: Loaded sensor specifications
        
    Returns:
        Configuration dictionary for the sensor.
    """
    sensors = specs.get('sensors', {})
    
    # Try exact match first
    if sensor_name in sensors:
        return sensors[sensor_name]
    
    # Try lowercase match
    sensor_name_lower = sensor_name.lower()
    if sensor_name_lower in sensors:
        return sensors[sensor_name_lower]
    
    # Return default configuration
    return sensors.get('default', {
        'expected_rate_hz': 100,
        'jitter_threshold_ms': 20,
        'gap_threshold_factor': 10.0
    })


def get_available_experiments(data_path: Optional[str] = None) -> Dict[str, str]:
    """
    Scan for available experiments in the data repository.
    
    Args:
        data_path: Path to data repository. If None, uses default.
        
    Returns:
        Dictionary mapping experiment names to paths.
    """
    if data_path is None:
        # Get absolute path relative to this file
        data_path = Path(__file__).parent.parent.parent / DATA_REPO_PATH
    else:
        data_path = Path(data_path)
    
    experiments = {}
    
    if not data_path.exists():
        warnings.warn(f"Data repository path not found: {data_path}")
        return experiments
    
    # Walk through directory tree
    for root, dirs, files in os.walk(data_path):
        root_path = Path(root)
        
        # Check if this directory contains GPS and IMU subdirectories
        has_gps = (root_path / GPS_SUBDIR).is_dir()
        has_imu = (root_path / IMU_SUBDIR).is_dir()
        
        if has_gps and has_imu:
            # This is a valid experiment directory
            relative_path = root_path.relative_to(data_path)
            display_name = str(relative_path).replace(os.sep, '/')
            experiments[display_name] = str(root_path)
            
            # Don't descend into GPS/IMU subdirectories
            dirs[:] = [d for d in dirs if d not in [GPS_SUBDIR, IMU_SUBDIR]]
    
    return dict(sorted(experiments.items()))


def load_gps_data(experiment_path: str) -> pd.DataFrame:
    """
    Load GPS data for an experiment.
    
    Args:
        experiment_path: Path to experiment directory
        
    Returns:
        DataFrame with GPS data including 'time_from_sync' column.
    """
    gps_dir = Path(experiment_path) / GPS_SUBDIR
    
    if not gps_dir.exists():
        warnings.warn(f"GPS directory not found: {gps_dir}")
        return pd.DataFrame()
    
    # Find GPS CSV file
    gps_files = list(gps_dir.glob("GPS_*.csv"))
    
    if not gps_files:
        warnings.warn(f"No GPS files found in {gps_dir}")
        return pd.DataFrame()
    
    # Use first file found (warn if multiple)
    if len(gps_files) > 1:
        warnings.warn(f"Multiple GPS files found in {gps_dir}, using {gps_files[0]}")
    
    try:
        df = pd.read_csv(gps_files[0])
        
        # Ensure time_from_sync column exists and is numeric
        if 'time_from_sync' in df.columns:
            df['time_from_sync'] = pd.to_numeric(df['time_from_sync'], errors='coerce')
            df.dropna(subset=['time_from_sync'], inplace=True)
            return df.sort_values('time_from_sync')
        elif 'Time' in df.columns:
            # Try to derive time_from_sync from Time column
            try:
                df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
                df.dropna(subset=['Time'], inplace=True)
                if not df.empty:
                    df['time_from_sync'] = (df['Time'] - df['Time'].min()).dt.total_seconds()
                    return df.sort_values('time_from_sync')
            except Exception as e:
                warnings.warn(f"Could not parse Time column: {e}")
        
        warnings.warn(f"No usable time column found in GPS data")
        return df
        
    except Exception as e:
        warnings.warn(f"Error loading GPS data: {e}")
        return pd.DataFrame()


def load_imu_data(experiment_path: str, sensor_name: str, 
                  measurement_type: str = 'accel') -> pd.DataFrame:
    """
    Load IMU data for a specific sensor and measurement type.
    
    Args:
        experiment_path: Path to experiment directory
        sensor_name: Name of IMU sensor (e.g., 'sensor_3')
        measurement_type: Type of measurement ('accel', 'gyro', 'angle', 'mag')
        
    Returns:
        DataFrame with IMU data including timestamp column.
    """
    # Get actual directory name for sensor
    actual_sensor_dir = SENSOR_DIR_MAP.get(sensor_name.lower(), sensor_name)
    imu_sensor_dir = Path(experiment_path) / IMU_SUBDIR / actual_sensor_dir
    
    if not imu_sensor_dir.exists():
        warnings.warn(f"IMU sensor directory not found: {imu_sensor_dir}")
        return pd.DataFrame()
    
    # Find measurement file
    pattern = f"{measurement_type}_*.csv"
    files = list(imu_sensor_dir.glob(pattern))
    
    if not files:
        warnings.warn(f"No {measurement_type} files found in {imu_sensor_dir}")
        return pd.DataFrame()
    
    # Use first file found
    if len(files) > 1:
        warnings.warn(f"Multiple {measurement_type} files found, using {files[0]}")
    
    try:
        df = pd.read_csv(files[0])
        
        # Handle different timestamp column names
        if 'time_from_sync' in df.columns:
            df['time_from_sync'] = pd.to_numeric(df['time_from_sync'], errors='coerce')
            df.dropna(subset=['time_from_sync'], inplace=True)
            return df.sort_values('time_from_sync')
        elif 't' in df.columns:
            # Rename 't' to 'time_from_sync' for consistency
            df['time_from_sync'] = pd.to_numeric(df['t'], errors='coerce')
            df.dropna(subset=['time_from_sync'], inplace=True)
            return df.sort_values('time_from_sync')
        else:
            warnings.warn(f"No timestamp column found in {files[0]}")
            return df
            
    except Exception as e:
        warnings.warn(f"Error loading IMU data: {e}")
        return pd.DataFrame()


def get_available_sensors(experiment_path: str) -> Dict[str, List[str]]:
    """
    Get available sensors and measurement types for an experiment.
    
    Args:
        experiment_path: Path to experiment directory
        
    Returns:
        Dictionary mapping sensor names to available measurement types.
    """
    imu_dir = Path(experiment_path) / IMU_SUBDIR
    available = {}
    
    # Check GPS
    gps_dir = Path(experiment_path) / GPS_SUBDIR
    if gps_dir.exists() and list(gps_dir.glob("GPS_*.csv")):
        available['gps'] = ['position']
    
    # Check IMU sensors
    if imu_dir.exists():
        for sensor_friendly, sensor_actual in SENSOR_DIR_MAP.items():
            sensor_dir = imu_dir / sensor_actual
            if sensor_dir.exists():
                measurements = []
                for mtype in IMU_MEASUREMENT_TYPES:
                    if list(sensor_dir.glob(f"{mtype}_*.csv")):
                        measurements.append(mtype)
                if measurements:
                    available[sensor_friendly] = measurements
    
    return available


def load_experiment_data(experiment_path: str, 
                        specs: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all sensor data for an experiment.
    
    Args:
        experiment_path: Path to experiment directory
        specs: Sensor specifications (loaded if not provided)
        
    Returns:
        Dictionary mapping sensor names to DataFrames.
    """
    if specs is None:
        specs = load_sensor_specs()
    
    data = {}
    available = get_available_sensors(experiment_path)
    
    # Load GPS data
    if 'gps' in available:
        gps_df = load_gps_data(experiment_path)
        if not gps_df.empty:
            data['gps'] = gps_df
    
    # Load IMU data (using first available measurement type for timestamp analysis)
    for sensor_name, measurements in available.items():
        if sensor_name != 'gps' and measurements:
            # Use first available measurement type
            imu_df = load_imu_data(experiment_path, sensor_name, measurements[0])
            if not imu_df.empty:
                data[sensor_name] = imu_df
    
    return data