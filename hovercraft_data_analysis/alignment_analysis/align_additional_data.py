#!/usr/bin/env python3
"""
Align additional sensor data (gyro, angle, mag) using timestamps from already-aligned data.
This script uses the aligned timestamps from accelerometer data to align other sensor data types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm


def align_sensor_data(sensor_timestamps, target_timestamps, sensor_data, tolerance_ms=2.5):
    """
    Align sensor data to target timestamps using nearest-neighbor matching.
    
    Args:
        sensor_timestamps: Original timestamps from sensor
        target_timestamps: Target timestamps to align to
        sensor_data: Data columns to align (N x M array)
        tolerance_ms: Maximum time difference allowed in milliseconds
        
    Returns:
        Aligned data array or None if alignment fails
    """
    # Convert tolerance to seconds
    tolerance_s = tolerance_ms / 1000.0
    
    # Find nearest neighbors using searchsorted
    indices = np.searchsorted(sensor_timestamps, target_timestamps, side='left')
    
    # Handle edge cases
    indices = np.clip(indices, 0, len(sensor_timestamps) - 1)
    
    # Check if we should use index or index-1
    use_prev = np.zeros(len(indices), dtype=bool)
    check_prev = indices > 0
    
    if np.any(check_prev):
        diff_current = np.abs(sensor_timestamps[indices[check_prev]] - target_timestamps[check_prev])
        diff_prev = np.abs(sensor_timestamps[indices[check_prev] - 1] - target_timestamps[check_prev])
        use_prev[check_prev] = diff_prev < diff_current
    
    indices[use_prev] -= 1
    
    # Calculate time differences
    time_diffs = np.abs(sensor_timestamps[indices] - target_timestamps)
    
    # Check which matches are within tolerance
    valid_mask = time_diffs <= tolerance_s
    
    # Create aligned data
    aligned_data = np.full((len(target_timestamps), sensor_data.shape[1]), np.nan)
    aligned_data[valid_mask] = sensor_data[indices[valid_mask]]
    
    return aligned_data, time_diffs * 1000  # Return diffs in ms


def process_experiment(experiment_name, base_data_path, aligned_data_path):
    """Process all sensor data for one experiment."""
    
    print(f"\nProcessing {experiment_name}...")
    
    # Find experiment path - search in both 02_Evaluation_Experiments and all_expts
    exp_path = None
    search_paths = [
        Path(base_data_path),
        Path(base_data_path).parent / "all_expts"
    ]
    
    for search_base in search_paths:
        if not search_base.exists():
            continue
        for path in search_base.rglob(experiment_name):
            if path.is_dir() and (path / "IMU").exists():
                exp_path = path
                break
            # Also check for direct sensor folders (morning structure)
            if path.is_dir() and (path / "Sensor_3").exists():
                exp_path = path
                break
        if exp_path:
            break
    
    if not exp_path:
        print(f"ERROR: Could not find experiment {experiment_name}")
        return False
    
    # Get aligned accelerometer data to extract reference timestamps
    aligned_csv_dir = aligned_data_path / f"{experiment_name}_csv"
    if not aligned_csv_dir.exists():
        print(f"ERROR: No aligned data found for {experiment_name}")
        return False
    
    # Read reference timestamps from any aligned sensor
    ref_sensor_file = aligned_csv_dir / "Sensor_3.csv"
    if not ref_sensor_file.exists():
        print(f"ERROR: No reference sensor data found")
        return False
    
    ref_df = pd.read_csv(ref_sensor_file)
    reference_timestamps = ref_df['t'].values
    
    print(f"Reference timestamps: {len(reference_timestamps)} samples")
    print(f"Time range: {reference_timestamps[0]:.1f} to {reference_timestamps[-1]:.1f} seconds")
    
    # Process each sensor
    sensors_to_process = ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']
    
    for sensor_name in sensors_to_process:
        print(f"\n  Processing {sensor_name}...")
        
        # Try both IMU subfolder and direct sensor folder
        sensor_path = exp_path / "IMU" / sensor_name
        if not sensor_path.exists():
            sensor_path = exp_path / sensor_name
        
        if not sensor_path.exists():
            print(f"    WARNING: Sensor path not found")
            continue
        
        # Load existing aligned data
        aligned_sensor_file = aligned_csv_dir / f"{sensor_name}.csv"
        if not aligned_sensor_file.exists():
            print(f"    WARNING: No aligned data found for {sensor_name}")
            continue
            
        aligned_df = pd.read_csv(aligned_sensor_file)
        
        # Use this sensor's timestamps as reference for alignment
        sensor_reference_timestamps = aligned_df['t'].values
        print(f"    Using {len(sensor_reference_timestamps)} timestamps from {sensor_name}")
        
        # Process each data type
        for data_type in ['gyro', 'angle', 'mag']:
            data_file = sensor_path / f"{data_type}_{experiment_name}.csv"
            if not data_file.exists():
                print(f"    WARNING: {data_type} data not found")
                continue
            
            # Load raw data
            raw_df = pd.read_csv(data_file)
            if 'time_from_sync' not in raw_df.columns:
                print(f"    WARNING: No time_from_sync column in {data_type} data")
                continue
            
            # Get data columns (exclude time columns)
            data_cols = [col for col in raw_df.columns if col not in ['t', 'time_from_sync']]
            
            # Align data using this sensor's specific timestamps
            aligned_data, time_diffs = align_sensor_data(
                raw_df['time_from_sync'].values,
                sensor_reference_timestamps,  # Use sensor-specific timestamps
                raw_df[data_cols].values,
                tolerance_ms=2.5 if sensor_name != 'Sensor_wb' else 5.0
            )
            
            # Add to existing dataframe
            # Check length compatibility
            if len(aligned_df) != len(aligned_data):
                print(f"    ERROR: Length mismatch - dataframe has {len(aligned_df)} rows, aligned data has {len(aligned_data)} rows")
                print(f"    Reference timestamps: {len(reference_timestamps)}")
                continue
            
            for i, col in enumerate(data_cols):
                col_name = f"{data_type}_{col}" if col in ['x', 'y', 'z'] else col
                aligned_df[col_name] = aligned_data[:, i]
            
            # Add time difference column for diagnostic
            aligned_df[f'{data_type}_time_diff_ms'] = time_diffs
            
            valid_count = np.sum(~np.isnan(aligned_data[:, 0]))
            print(f"    {data_type}: {valid_count}/{len(sensor_reference_timestamps)} samples aligned")
        
        # Save updated dataframe
        aligned_df.to_csv(aligned_sensor_file, index=False)
        print(f"    Updated {aligned_sensor_file.name}")
    
    return True


def main():
    """Process all experiments."""
    
    base_data_path = Path(__file__).parent.parent.parent / "02_Evaluation_Experiments"
    aligned_data_path = Path(__file__).parent / "aligned_data"
    
    experiments = [
        "011_Static_stbd_1",
        "012_Static_port_1",
        "013_Static_port_2",
        "014_Static_stbd_2"
    ]
    
    success_count = 0
    
    for exp in experiments:
        if process_experiment(exp, base_data_path, aligned_data_path):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Alignment complete: {success_count}/{len(experiments)} experiments processed")
    print(f"{'='*60}")
    
    if success_count < len(experiments):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())