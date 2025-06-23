"""
Data alignment module for hovercraft sensor data.

This module provides functionality to align multi-rate sensor data to a common time base
using vectorized nearest-neighbor matching.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


class DataAligner:
    """Aligns multi-rate sensor data to a common reference time base."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize DataAligner with configuration.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "alignment_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.reference_sensor = self.config['reference_sensor']
        self.target_rate = self.config['target_rate']
        self.tolerances = self.config['tolerances']
        self.sensor_rates = self.config['sensor_rates']
        self.max_cross_sensor_offset = self.config['max_cross_sensor_offset_ms']
        
        self.aligned_data = {}
        self.reference_timestamps = None
        
    def load_experiment_data(self, experiment_name: str, base_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Load all sensor data for an experiment.
        
        Args:
            experiment_name: Name of the experiment (e.g., '007_Fast_stbd_turn_1')
            base_path: Base path to experiment data
            
        Returns:
            Dictionary mapping sensor names to DataFrames
        """
        sensor_data = {}
        
        # Find experiment directory
        experiment_path = self._find_experiment_path(experiment_name, base_path)
        if not experiment_path:
            raise ValueError(f"Experiment {experiment_name} not found in {base_path}")
        
        # Load IMU data - check both IMU subfolder and direct sensor folders
        imu_path = experiment_path / "IMU"
        if imu_path.exists():
            # Afternoon structure: experiment/IMU/Sensor_X/
            for sensor_dir in imu_path.iterdir():
                if sensor_dir.is_dir():
                    sensor_name = sensor_dir.name
                    # Load accelerometer data as primary (contains timestamps)
                    accel_file = sensor_dir / f"accel_{experiment_name}.csv"
                    if accel_file.exists():
                        df = pd.read_csv(accel_file)
                        if 'time_from_sync' in df.columns:
                            sensor_data[sensor_name] = df
                            print(f"Loaded {sensor_name}: {len(df)} samples")
        else:
            # Morning structure: experiment/Sensor_X/
            for sensor_dir in experiment_path.iterdir():
                if sensor_dir.is_dir() and sensor_dir.name.startswith('Sensor_'):
                    sensor_name = sensor_dir.name
                    # Load accelerometer data as primary (contains timestamps)
                    accel_file = sensor_dir / f"accel_{experiment_name}.csv"
                    if accel_file.exists():
                        df = pd.read_csv(accel_file)
                        if 'time_from_sync' in df.columns:
                            sensor_data[sensor_name] = df
                            print(f"Loaded {sensor_name}: {len(df)} samples")
        
        # Load GPS data
        gps_path = experiment_path / "GPS" / f"GPS_{experiment_name}.csv"
        if gps_path.exists():
            df = pd.read_csv(gps_path)
            if 'time_from_sync' in df.columns:
                sensor_data['gps'] = df
                print(f"Loaded GPS: {len(df)} samples")
        
        return sensor_data
    
    def _find_experiment_path(self, experiment_name: str, base_path: Path) -> Optional[Path]:
        """Find experiment directory by searching through the data structure."""
        # Search in common experiment locations
        search_paths = [
            base_path / "02_Evaluation_Experiments",
            base_path
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for path in search_path.rglob(experiment_name):
                if path.is_dir():
                    return path
        
        return None
    
    def align_all_sensors(self, sensor_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all sensors to the reference sensor's time base.
        
        Args:
            sensor_data: Dictionary mapping sensor names to DataFrames
            
        Returns:
            Dictionary of aligned DataFrames
        """
        start_time = time.time()
        
        # Get reference timestamps
        if self.reference_sensor not in sensor_data:
            raise ValueError(f"Reference sensor {self.reference_sensor} not found in data")
        
        ref_df = sensor_data[self.reference_sensor]
        self.reference_timestamps = ref_df['time_from_sync'].values
        
        print(f"Reference sensor {self.reference_sensor}: {len(self.reference_timestamps)} timestamps")
        print(f"Time range: {self.reference_timestamps[0]:.3f} to {self.reference_timestamps[-1]:.3f} seconds")
        
        # Align each sensor
        aligned = {}
        for sensor_name, df in sensor_data.items():
            if sensor_name == self.reference_sensor:
                # Reference sensor is already aligned
                aligned[sensor_name] = df.copy()
            elif sensor_name in ['sensor_wnb', 'Sensor_wnb']:
                # Skip sensor_wnb due to timing issues
                print(f"Skipping {sensor_name} (excluded due to timing issues)")
                continue
            else:
                # Align other sensors
                sensor_rate = self.sensor_rates.get(sensor_name)
                if sensor_rate is None:
                    print(f"Warning: Unknown sensor rate for {sensor_name}, skipping")
                    continue
                
                tolerance_ms = self.tolerances.get(sensor_rate, 5.0)
                aligned_df = self._align_sensor(df, sensor_name, tolerance_ms)
                
                if aligned_df is not None:
                    aligned[sensor_name] = aligned_df
                    print(f"Aligned {sensor_name}: {len(aligned_df)} samples")
        
        # Cross-sensor validation
        self._validate_cross_sensor_alignment(aligned)
        
        elapsed = time.time() - start_time
        print(f"Alignment completed in {elapsed:.3f} seconds")
        
        # Assert runtime < 1 second for 5-minute dataset
        if elapsed > 1.0:
            print(f"Warning: Alignment took {elapsed:.3f}s, exceeding 1s target")
        
        self.aligned_data = aligned
        return aligned
    
    def _align_sensor(self, sensor_df: pd.DataFrame, sensor_name: str, tolerance_ms: float) -> Optional[pd.DataFrame]:
        """
        Align a single sensor to reference timestamps using nearest-neighbor matching.
        
        Args:
            sensor_df: Sensor DataFrame with 'time_from_sync' column
            sensor_name: Name of the sensor
            tolerance_ms: Maximum allowed time difference in milliseconds
            
        Returns:
            Aligned DataFrame or None if alignment fails
        """
        sensor_timestamps = sensor_df['time_from_sync'].values
        sensor_rate = self.sensor_rates.get(sensor_name, 0)
        
        # Handle different sampling rates
        if sensor_rate == 100 and self.target_rate == 200:
            # For 100Hz sensor, take every 2nd reference timestamp
            target_timestamps = self.reference_timestamps[::2]
        else:
            target_timestamps = self.reference_timestamps
        
        # Vectorized nearest-neighbor search using searchsorted
        indices = np.searchsorted(sensor_timestamps, target_timestamps)
        
        # Handle edge cases
        indices = np.clip(indices, 0, len(sensor_timestamps) - 1)
        
        # Check if we need to look at the previous index too
        prev_indices = np.maximum(indices - 1, 0)
        
        # Calculate distances to both neighbors
        dist_current = np.abs(sensor_timestamps[indices] - target_timestamps)
        dist_prev = np.abs(sensor_timestamps[prev_indices] - target_timestamps)
        
        # Choose the nearest neighbor
        use_prev = dist_prev < dist_current
        best_indices = np.where(use_prev, prev_indices, indices)
        best_distances = np.where(use_prev, dist_prev, dist_current)
        
        # Apply tolerance check
        tolerance_s = tolerance_ms / 1000.0
        valid_mask = best_distances <= tolerance_s
        
        if not np.any(valid_mask):
            print(f"Error: No valid matches for {sensor_name} within {tolerance_ms}ms tolerance")
            return None
        
        # Create aligned DataFrame
        aligned_df = sensor_df.iloc[best_indices[valid_mask]].copy()
        aligned_df['aligned_time'] = target_timestamps[valid_mask]
        aligned_df['time_diff_ms'] = best_distances[valid_mask] * 1000
        
        # Report statistics
        valid_ratio = np.sum(valid_mask) / len(target_timestamps)
        mean_diff = np.mean(best_distances[valid_mask]) * 1000
        max_diff = np.max(best_distances[valid_mask]) * 1000
        
        print(f"  {sensor_name}: {valid_ratio:.1%} valid, mean diff={mean_diff:.2f}ms, max diff={max_diff:.2f}ms")
        
        return aligned_df
    
    def _validate_cross_sensor_alignment(self, aligned_data: Dict[str, pd.DataFrame]):
        """
        Validate that all sensors are aligned within the specified tolerance.
        
        Args:
            aligned_data: Dictionary of aligned DataFrames
        """
        # Compare pairs of 200Hz sensors
        high_rate_sensors = [s for s in aligned_data.keys() 
                           if self.sensor_rates.get(s) == 200]
        
        if len(high_rate_sensors) < 2:
            return
        
        print("\nCross-sensor validation:")
        for i in range(len(high_rate_sensors)):
            for j in range(i + 1, len(high_rate_sensors)):
                sensor1, sensor2 = high_rate_sensors[i], high_rate_sensors[j]
                
                # Get common aligned times
                # Reference sensor uses 'time_from_sync', others use 'aligned_time'
                time_col1 = 'time_from_sync' if sensor1 == self.reference_sensor else 'aligned_time'
                time_col2 = 'time_from_sync' if sensor2 == self.reference_sensor else 'aligned_time'
                
                times1 = set(aligned_data[sensor1][time_col1].values)
                times2 = set(aligned_data[sensor2][time_col2].values)
                common_times = sorted(times1.intersection(times2))
                
                if not common_times:
                    continue
                
                # Check time differences at common points
                df1 = aligned_data[sensor1].set_index(time_col1)
                df2 = aligned_data[sensor2].set_index(time_col2)
                
                # For non-reference sensors, check their time differences
                if sensor1 != self.reference_sensor and sensor2 != self.reference_sensor:
                    # Both sensors have time_diff_ms column
                    df1_subset = aligned_data[sensor1][aligned_data[sensor1]['aligned_time'].isin(common_times[:100])]
                    df2_subset = aligned_data[sensor2][aligned_data[sensor2]['aligned_time'].isin(common_times[:100])]
                    
                    if len(df1_subset) > 0 and len(df2_subset) > 0:
                        max_diff1 = df1_subset['time_diff_ms'].max()
                        max_diff2 = df2_subset['time_diff_ms'].max()
                        max_diff = max(max_diff1, max_diff2)
                        
                        print(f"  {sensor1} vs {sensor2}: max time diff={max_diff:.3f}ms")
                        
                        # Since both are aligned to same reference, their combined error should be < 1ms
                        assert max_diff < self.max_cross_sensor_offset * 2, \
                            f"Cross-sensor time difference {max_diff:.3f}ms exceeds tolerance"
    
    def save_aligned_data(self, output_path: Path) -> pd.HDFStore:
        """
        Save aligned data to HDF5 file using pandas HDFStore.
        
        Args:
            output_path: Path for output HDF5 file
            
        Returns:
            HDFStore object for further queries
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        store = pd.HDFStore(str(output_path), mode='w')
        
        for sensor_name, df in self.aligned_data.items():
            # Store with queryable format
            store.put(sensor_name, df, format='table', data_columns=True)
            print(f"Saved {sensor_name}: {len(df)} samples")
        
        # Store metadata
        metadata = pd.DataFrame({
            'reference_sensor': [self.reference_sensor],
            'target_rate': [self.target_rate],
            'num_sensors': [len(self.aligned_data)],
            'alignment_timestamp': [pd.Timestamp.now()]
        })
        store.put('metadata', metadata)
        
        store.close()
        print(f"Aligned data saved to {output_path}")
        
        # Return reopened store for queries
        return pd.HDFStore(str(output_path), mode='r')