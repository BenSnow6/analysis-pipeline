"""
Data I/O operations for RPM estimation.

This module handles loading sensor data, configuration files,
and saving results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from .logging_config import get_logger, ProcessingError

logger = get_logger("io")


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


def load_orientation_config(config_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """
    Load orientation configuration with rotation matrices.
    
    Args:
        config_path: Path to orientation_config.yaml 
                    (defaults to searching common locations)
        
    Returns:
        Dictionary mapping sensor IDs to rotation matrices (R_bs)
        
    Raises:
        FileNotFoundError: If orientation config not found
        ValueError: If rotation matrices are invalid
    """
    if config_path is None:
        # Search for orientation config in common locations
        module_path = Path(__file__).parent
        search_paths = [
            module_path.parent.parent / 'hovercraft_data_analysis' / 'alignment_analysis' / 'orientation_config.yaml',
            module_path.parent.parent / 'config' / 'orientation_config.yaml',
            module_path / 'orientation_config.yaml'
        ]
        
        for path in search_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(
                "orientation_config.yaml not found. Searched: " + 
                ", ".join(str(p) for p in search_paths)
            )
    
    try:
        with open(config_path, 'r') as f:
            orientation_data = yaml.safe_load(f)
    except Exception as e:
        logger.error(
            f"Failed to load orientation config from {config_path}",
            error_type=ProcessingError.CONFIG
        )
        raise
    
    # Extract rotation matrices
    rotation_matrices = {}
    
    for sensor_id, sensor_data in orientation_data.items():
        if isinstance(sensor_data, dict) and 'R_bs' in sensor_data:
            R_bs = np.array(sensor_data['R_bs'])
            
            # Validate rotation matrix
            if R_bs.shape != (3, 3):
                raise ValueError(f"Invalid rotation matrix shape for {sensor_id}: {R_bs.shape}")
            
            # Check if it's a valid rotation matrix (orthogonal with det=1)
            if not np.allclose(np.linalg.det(R_bs), 1.0, atol=1e-6):
                logger.warning(
                    f"Rotation matrix for {sensor_id} has determinant {np.linalg.det(R_bs):.6f} (expected 1.0)"
                )
            
            rotation_matrices[sensor_id] = R_bs
            logger.debug(f"Loaded rotation matrix for {sensor_id}")
    
    logger.info(f"Loaded orientation data for {len(rotation_matrices)} sensors from {config_path}")
    return rotation_matrices


def apply_body_rotation(accel_data: Union[pd.DataFrame, np.ndarray], 
                       R_bs: np.ndarray,
                       accel_cols: Tuple[str, str, str] = ('x', 'y', 'z')) -> np.ndarray:
    """
    Apply body-frame rotation to acceleration data.
    
    Args:
        accel_data: DataFrame or array with acceleration data
        R_bs: 3x3 rotation matrix (body to sensor transform)
        accel_cols: Column names for x, y, z accelerations
        
    Returns:
        Rotated accelerations as Nx3 array
    """
    # Extract acceleration vectors
    if isinstance(accel_data, pd.DataFrame):
        accel_sensor = accel_data[list(accel_cols)].values
    else:
        accel_sensor = accel_data
    
    # Validate dimensions
    if accel_sensor.shape[1] != 3:
        raise ValueError(f"Expected 3D acceleration data, got shape {accel_sensor.shape}")
    
    # Apply rotation: a_body = R_bs^T @ a_sensor
    # Note: R_bs transforms from body to sensor, so we use transpose for sensor to body
    accel_body = accel_sensor @ R_bs.T
    
    return accel_body


def load_aligned_data(experiment: str, session: str, sensor_id: str,
                     base_path: Optional[Path] = None,
                     apply_rotation: bool = True) -> pd.DataFrame:
    """
    Load aligned sensor data with optional rotation to body frame.
    
    Args:
        experiment: Experiment name
        session: 'morning' or 'afternoon'
        sensor_id: Sensor identifier
        base_path: Base directory for data
        apply_rotation: Whether to apply body-frame rotation
        
    Returns:
        DataFrame with aligned data, optionally rotated to body frame
    """
    # Load raw sensor data
    df = load_sensor_data(experiment, session, sensor_id, base_path)
    
    if apply_rotation:
        try:
            # Load rotation matrices
            rotation_matrices = load_orientation_config()
            
            if sensor_id in rotation_matrices:
                # Apply rotation
                R_bs = rotation_matrices[sensor_id]
                accel_body = apply_body_rotation(df, R_bs)
                
                # Add body-frame columns
                df['x_body'] = accel_body[:, 0]
                df['y_body'] = accel_body[:, 1]
                df['z_body'] = accel_body[:, 2]
                
                logger.info(
                    f"Applied body-frame rotation to {sensor_id}",
                    sensor=sensor_id,
                    processing_step="rotation"
                )
            else:
                logger.warning(
                    f"No rotation matrix found for {sensor_id}, using sensor frame",
                    sensor=sensor_id,
                    error_type=ProcessingError.RECOVERABLE
                )
                # Copy sensor frame as body frame
                df['x_body'] = df['x']
                df['y_body'] = df['y']
                df['z_body'] = df['z']
                
        except Exception as e:
            logger.error(
                f"Failed to apply rotation for {sensor_id}: {str(e)}",
                sensor=sensor_id,
                error_type=ProcessingError.RECOVERABLE
            )
            # Fallback to sensor frame
            df['x_body'] = df['x']
            df['y_body'] = df['y']
            df['z_body'] = df['z']
    
    return df


def save_processed_data(df: pd.DataFrame, output_path: Path,
                       compression: str = 'snappy',
                       schema_version: str = '1.0') -> None:
    """
    Save processed data as Parquet file with metadata.
    
    Args:
        df: DataFrame with processed data
        output_path: Path to save Parquet file
        compression: Compression algorithm
        schema_version: Schema version for compatibility tracking
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata - PyArrow expects bytes for metadata values
    metadata = {
        b'schema_version': str(schema_version).encode('utf-8'),
        b'processing_timestamp': pd.Timestamp.now().isoformat().encode('utf-8'),
        b'columns': str(list(df.columns)).encode('utf-8'),
        b'shape': f"{df.shape[0]} x {df.shape[1]}".encode('utf-8')
    }
    
    # Convert to PyArrow table with metadata
    table = pa.Table.from_pandas(df)
    table = table.replace_schema_metadata(metadata)
    
    # Write Parquet file
    pq.write_table(table, output_path, compression=compression)
    
    logger.info(
        f"Saved processed data to {output_path}",
        rows=df.shape[0],
        columns=df.shape[1],
        compression=compression
    )