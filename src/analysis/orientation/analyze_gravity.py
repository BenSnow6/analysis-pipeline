#!/usr/bin/env python3
"""
Analyze gravity measurements to understand the small magnitude issue.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from src.core.paths import ORIENTATION_CONFIG_FILE

def analyze_gravity_measurements():
    """Analyze raw accelerometer data to check gravity magnitude."""
    
    # Load some raw data from a static experiment
    base_path = Path("/mnt/c/Users/ben/Documents/EngD/09 Data collection/01_analysis_pipeline/analysis-pipeline")
    
    # Check a setup experiment (should be static)
    experiment = "002_Setup"
    sensor = "Sensor_3"
    
    # Try to load from original data
    possible_paths = [
        base_path / "all_expts" / "afternoon" / "Experiments" / experiment / "IMU" / sensor / f"accel_{experiment}.csv",
        base_path / "all_expts" / "morning" / "Experiments" / experiment / "IMU" / sensor / f"accel_{experiment}.csv",
        base_path / "data/raw" / "1a_1_Minimum_Radius_Turn" / "morning" / experiment / sensor / f"accel_{experiment}.csv",
        base_path / "data/raw" / "1a_1_Minimum_Radius_Turn" / "morning" / experiment / "IMU" / sensor / f"accel_{experiment}.csv"
    ]
    
    accel_data = None
    for path in possible_paths:
        if path.exists():
            print(f"Found data at: {path}")
            df = pd.read_csv(path)
            print(f"Columns: {df.columns.tolist()}")
            print(f"Shape: {df.shape}")
            
            # Extract acceleration columns
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                accel_data = df[['x', 'y', 'z']].values
            elif 'accel_x' in df.columns:
                accel_data = df[['accel_x', 'accel_y', 'accel_z']].values
            break
    
    if accel_data is None:
        print("Could not find accelerometer data")
        return
        
    # Analyze the data
    print(f"\n=== Raw Accelerometer Analysis ===")
    print(f"Data shape: {accel_data.shape}")
    
    # Take first 1000 samples (should be static)
    static_data = accel_data[:1000]
    
    # Calculate statistics
    mean_accel = np.mean(static_data, axis=0)
    std_accel = np.std(static_data, axis=0)
    magnitude = np.linalg.norm(mean_accel)
    
    print(f"\nMean acceleration: {mean_accel}")
    print(f"Std deviation: {std_accel}")
    print(f"Magnitude: {magnitude:.3f} m/s²")
    print(f"Expected gravity: 9.80665 m/s²")
    print(f"Ratio: {magnitude / 9.80665:.3f}")
    
    # Check if data might be in g's instead of m/s²
    if magnitude < 2.0:
        print(f"\n⚠️ Data appears to be in g's, not m/s²!")
        print(f"Magnitude in g's: {magnitude:.3f} g")
        print(f"Converted to m/s²: {magnitude * 9.80665:.3f} m/s²")
    
    # Check individual axes
    print(f"\n=== Per-axis Analysis ===")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"{axis}-axis: mean={mean_accel[i]:.3f}, std={std_accel[i]:.3f}")
    
    # Find which axis has the largest magnitude
    max_axis = np.argmax(np.abs(mean_accel))
    axis_names = ['X', 'Y', 'Z']
    print(f"\nLargest magnitude on {axis_names[max_axis]}-axis: {mean_accel[max_axis]:.3f}")
    
    # Check if it's inverted
    if mean_accel[max_axis] < 0:
        print("⚠️ Gravity appears to be negative - sensor might be inverted!")
    
    # Now check the rotation matrix expectation
    config_path = ORIENTATION_CONFIG_FILE
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n=== Configuration Analysis ===")
    expected_gravity = config['physics']['gravity_body_frame']
    print(f"Expected gravity in body frame: {expected_gravity}")
    print(f"This means gravity should point in the {['X', 'Y', 'Z'][expected_gravity.index(9.80665)]} direction")
    
    # Check sensor mounting
    sensor_config = config['sensors'][sensor]
    print(f"\nSensor {sensor} expected axes:")
    print(f"  X: {sensor_config['expected_axes']['x_direction']}")
    print(f"  Y: {sensor_config['expected_axes']['y_direction']}")
    print(f"  Z: {sensor_config['expected_axes']['z_direction']}")

if __name__ == "__main__":
    analyze_gravity_measurements()