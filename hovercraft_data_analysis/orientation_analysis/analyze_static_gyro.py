#!/usr/bin/env python3
"""
Analyze gyroscope data from static experiments to understand why static detector isn't finding static segments.
This script loads gyro data, calculates angular velocity magnitude, and prints statistics.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path


def analyze_gyro_data(gyro_file, threshold=0.05):
    """
    Analyze gyroscope data from a static experiment.
    
    Parameters:
    -----------
    gyro_file : str
        Path to the gyro CSV file
    threshold : float
        Threshold for static detection in rad/s (default: 0.05)
    """
    print(f"\nAnalyzing gyro data from: {gyro_file}")
    print(f"Static detection threshold: {threshold} rad/s")
    print("-" * 80)
    
    # Load the data
    try:
        df = pd.read_csv(gyro_file)
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Check if data has gyro columns (assuming columns like gx, gy, gz or similar)
    gyro_cols = []
    for col in df.columns:
        if any(x in col.lower() for x in ['gx', 'gy', 'gz', 'gyro', 'angular']):
            gyro_cols.append(col)
    
    if len(gyro_cols) < 3:
        # Try alternative naming
        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
            gyro_cols = ['x', 'y', 'z']
        else:
            print(f"Could not find gyro columns. Available columns: {list(df.columns)}")
            return
    
    print(f"\nUsing gyro columns: {gyro_cols}")
    
    # Extract gyro data
    gx = df[gyro_cols[0]].values
    gy = df[gyro_cols[1]].values
    gz = df[gyro_cols[2]].values
    
    # Calculate magnitude of angular velocity
    gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
    
    # Calculate statistics
    print("\n--- Angular Velocity Statistics ---")
    print(f"Mean magnitude: {np.mean(gyro_magnitude):.6f} rad/s")
    print(f"Std magnitude: {np.std(gyro_magnitude):.6f} rad/s")
    print(f"Min magnitude: {np.min(gyro_magnitude):.6f} rad/s")
    print(f"Max magnitude: {np.max(gyro_magnitude):.6f} rad/s")
    print(f"Median magnitude: {np.median(gyro_magnitude):.6f} rad/s")
    
    # Component-wise statistics
    print("\n--- Component-wise Statistics ---")
    for i, (component, data) in enumerate(zip(['X', 'Y', 'Z'], [gx, gy, gz])):
        print(f"\n{component}-axis:")
        print(f"  Mean: {np.mean(data):.6f} rad/s")
        print(f"  Std: {np.std(data):.6f} rad/s")
        print(f"  Min: {np.min(data):.6f} rad/s")
        print(f"  Max: {np.max(data):.6f} rad/s")
    
    # Threshold analysis
    print(f"\n--- Threshold Analysis ---")
    below_threshold = np.sum(gyro_magnitude < threshold)
    percentage_below = (below_threshold / len(gyro_magnitude)) * 100
    
    print(f"Samples below {threshold} rad/s: {below_threshold}/{len(gyro_magnitude)} ({percentage_below:.2f}%)")
    
    # Try different thresholds
    print("\n--- Testing Different Thresholds ---")
    test_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    for test_threshold in test_thresholds:
        below = np.sum(gyro_magnitude < test_threshold)
        percent = (below / len(gyro_magnitude)) * 100
        print(f"Threshold {test_threshold:.2f} rad/s: {below}/{len(gyro_magnitude)} samples ({percent:.2f}%)")
    
    # Check for periods of low activity
    print("\n--- Checking for Static Periods ---")
    # Use a sliding window to find consecutive static samples
    window_size = 100  # samples
    static_windows = []
    
    for i in range(len(gyro_magnitude) - window_size):
        window = gyro_magnitude[i:i+window_size]
        if np.max(window) < threshold:
            static_windows.append(i)
    
    if static_windows:
        print(f"Found {len(static_windows)} windows of {window_size} samples where all values < {threshold} rad/s")
        # Find continuous segments
        segments = []
        if static_windows:
            start = static_windows[0]
            for i in range(1, len(static_windows)):
                if static_windows[i] != static_windows[i-1] + 1:
                    segments.append((start, static_windows[i-1]))
                    start = static_windows[i]
            segments.append((start, static_windows[-1]))
            
            print(f"\nFound {len(segments)} continuous static segments:")
            for i, (start, end) in enumerate(segments[:5]):  # Show first 5
                duration = (end - start) / 100.0  # Assuming 100Hz
                print(f"  Segment {i+1}: samples {start}-{end} (duration: {duration:.2f}s)")
    else:
        print(f"No windows of {window_size} samples found where all values < {threshold} rad/s")
        print("This suggests the data may be too noisy or the threshold is too low.")


def main():
    # Default to analyzing Sensor_3 from experiment 010
    base_path = Path(__file__).parent.parent.parent
    default_gyro_file = base_path / "all_expts/afternoon/Experiments/010_Waiting_for_static_turns/IMU/Sensor_3/gyro_010_Waiting_for_static_turns.csv"
    
    if len(sys.argv) > 1:
        gyro_file = sys.argv[1]
    else:
        gyro_file = str(default_gyro_file)
    
    if not os.path.exists(gyro_file):
        print(f"File not found: {gyro_file}")
        print("\nSearching for available static experiment gyro files...")
        # Search for static experiment files
        search_patterns = ["*Waiting*", "*Static*", "*static*"]
        found_files = []
        
        for pattern in search_patterns:
            for file in base_path.rglob(f"*gyro*{pattern}*.csv"):
                if "gyro" in file.name:
                    found_files.append(str(file))
        
        if found_files:
            print("\nFound the following gyro files from static experiments:")
            for i, file in enumerate(found_files[:10]):  # Show first 10
                print(f"{i+1}. {file}")
            print("\nPlease run the script with one of these files as argument.")
        return
    
    # Analyze the data
    analyze_gyro_data(gyro_file)
    
    # Also analyze other sensors for comparison
    print("\n" + "="*80)
    print("ANALYZING OTHER SENSORS FOR COMPARISON")
    print("="*80)
    
    gyro_dir = Path(gyro_file).parent.parent
    for sensor_dir in gyro_dir.glob("Sensor_*"):
        sensor_gyro = sensor_dir / Path(gyro_file).name
        if sensor_gyro.exists() and str(sensor_gyro) != gyro_file:
            print(f"\n{'='*40}")
            print(f"Sensor: {sensor_dir.name}")
            print(f"{'='*40}")
            analyze_gyro_data(str(sensor_gyro))


if __name__ == "__main__":
    main()