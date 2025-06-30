"""
Common I/O utilities for the hovercraft analysis pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load CSV data with consistent defaults.

    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.ParserError: If CSV parsing fails
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # Set sensible defaults
    defaults = {
        "index_col": False,
        "parse_dates": False,  # Let caller specify date columns
    }
    defaults.update(kwargs)

    try:
        return pd.read_csv(filepath, **defaults)
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV {filepath}: {e}")
        raise


def save_csv_data(data: pd.DataFrame, filepath: Union[str, Path], **kwargs):
    """
    Save DataFrame to CSV with consistent defaults.

    Args:
        data: DataFrame to save
        filepath: Output path
        **kwargs: Additional arguments passed to to_csv
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    defaults = {
        "index": False,
        "float_format": "%.6f",
    }
    defaults.update(kwargs)

    data.to_csv(filepath, **defaults)
    logger.info(f"Saved CSV to {filepath}")


def load_hdf5_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from HDF5 file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        Dictionary with dataset names as keys

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    data = {}
    with h5py.File(filepath, "r") as f:

        def _extract_data(name, obj):
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[:]
            elif isinstance(obj, h5py.Group):
                # Store attributes if any
                if obj.attrs:
                    data[f"{name}_attrs"] = dict(obj.attrs)

        f.visititems(_extract_data)

    return data


def save_hdf5_data(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    compression: str = "gzip",
    compression_opts: int = 4,
):
    """
    Save data to HDF5 file.

    Args:
        data: Dictionary of datasets to save
        filepath: Output path
        compression: Compression algorithm
        compression_opts: Compression level (1-9)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                # Save DataFrame columns separately
                group = f.create_group(key)
                for col in value.columns:
                    group.create_dataset(
                        col,
                        data=value[col].values,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                # Save column names and index
                group.attrs["columns"] = list(value.columns)
                if value.index.name:
                    group.attrs["index_name"] = value.index.name
            elif isinstance(value, (np.ndarray, list)):
                f.create_dataset(
                    key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            elif isinstance(value, dict) and key.endswith("_attrs"):
                # Handle attributes
                group_name = key.replace("_attrs", "")
                if group_name in f:
                    for attr_key, attr_val in value.items():
                        f[group_name].attrs[attr_key] = attr_val
            else:
                # Store as attribute if it's a simple type
                f.attrs[key] = value

    logger.info(f"Saved HDF5 to {filepath}")


def load_experiment_data(
    experiment_name: str, time_of_day: str = "morning"
) -> Dict[str, pd.DataFrame]:
    """
    Load all data for an experiment.

    Args:
        experiment_name: Name of the experiment
        time_of_day: "morning" or "afternoon"

    Returns:
        Dictionary with sensor names as keys and DataFrames as values
    """
    from .paths import get_experiment_path

    exp_path = get_experiment_path(experiment_name, time_of_day)
    data = {}

    # Load GPS data
    gps_path = exp_path / "GPS" / f"GPS_{experiment_name}.csv"
    if gps_path.exists():
        data["gps"] = load_csv_data(gps_path)

    # Load IMU data
    imu_path = exp_path / "IMU"
    if imu_path.exists():
        for sensor_dir in imu_path.iterdir():
            if sensor_dir.is_dir():
                sensor_name = sensor_dir.name

                # Load each data type
                for data_type in ["accel", "gyro", "mag", "angle"]:
                    file_path = sensor_dir / f"{data_type}_{experiment_name}.csv"
                    if file_path.exists():
                        key = f"{sensor_name}_{data_type}"
                        data[key] = load_csv_data(file_path)

    logger.info(f"Loaded {len(data)} datasets for experiment {experiment_name}")
    return data


def validate_dataframe(
    df: pd.DataFrame, required_columns: List[str], name: str = "DataFrame"
) -> bool:
    """
    Validate that a DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages

    Returns:
        True if valid, False otherwise
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        logger.error(f"{name} missing required columns: {missing}")
        return False
    return True


def ensure_timestamp_index(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Ensure DataFrame has a datetime index.

    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with datetime index
    """
    if timestamp_col in df.columns:
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df.set_index(timestamp_col, inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame does not have a timestamp column or datetime index")

    return df
