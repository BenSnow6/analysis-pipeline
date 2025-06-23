#!/usr/bin/env python3
"""
Simple analysis of gyroscope data from static experiments without external dependencies.
This script loads gyro data, calculates angular velocity magnitude, and prints statistics.
"""

import csv
import math
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
        with open(gyro_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = list(reader)
        
        print(f"Loaded {len(data)} samples")
        print(f"Columns: {headers}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Find gyro columns
    gyro_indices = []
    for i, col in enumerate(headers):
        if any(x in col.lower() for x in ['gx', 'gy', 'gz', 'gyro', 'angular']):
            gyro_indices.append(i)
    
    if len(gyro_indices) < 3:
        # Try alternative naming
        if 'x' in headers and 'y' in headers and 'z' in headers:
            gyro_indices = [headers.index('x'), headers.index('y'), headers.index('z')]
        else:
            print(f"Could not find gyro columns. Available columns: {headers}")
            return
    
    print(f"\nUsing gyro columns: {[headers[i] for i in gyro_indices[:3]]}")
    
    # Extract gyro data and calculate magnitude
    gyro_magnitudes = []
    gx_values = []
    gy_values = []
    gz_values = []
    
    for row in data:
        try:
            gx = float(row[gyro_indices[0]])
            gy = float(row[gyro_indices[1]])
            gz = float(row[gyro_indices[2]])
            
            gx_values.append(gx)
            gy_values.append(gy)
            gz_values.append(gz)
            
            magnitude = math.sqrt(gx**2 + gy**2 + gz**2)
            gyro_magnitudes.append(magnitude)
        except (ValueError, IndexError):
            continue
    
    if not gyro_magnitudes:
        print("No valid gyro data found")
        return
    
    # Calculate statistics
    def calc_stats(values):
        n = len(values)
        if n == 0:
            return None, None, None, None, None
        
        mean = sum(values) / n
        sorted_values = sorted(values)
        median = sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        min_val = min(values)
        max_val = max(values)
        
        # Standard deviation
        variance = sum((x - mean) ** 2 for x in values) / n
        std = math.sqrt(variance)
        
        return mean, std, min_val, max_val, median
    
    mean_mag, std_mag, min_mag, max_mag, median_mag = calc_stats(gyro_magnitudes)
    
    print("\n--- Angular Velocity Statistics ---")
    print(f"Mean magnitude: {mean_mag:.6f} rad/s")
    print(f"Std magnitude: {std_mag:.6f} rad/s")
    print(f"Min magnitude: {min_mag:.6f} rad/s")
    print(f"Max magnitude: {max_mag:.6f} rad/s")
    print(f"Median magnitude: {median_mag:.6f} rad/s")
    
    # Component-wise statistics
    print("\n--- Component-wise Statistics ---")
    for axis, values in [('X', gx_values), ('Y', gy_values), ('Z', gz_values)]:
        mean_val, std_val, min_val, max_val, _ = calc_stats(values)
        print(f"\n{axis}-axis:")
        print(f"  Mean: {mean_val:.6f} rad/s")
        print(f"  Std: {std_val:.6f} rad/s")
        print(f"  Min: {min_val:.6f} rad/s")
        print(f"  Max: {max_val:.6f} rad/s")
    
    # Threshold analysis
    print(f"\n--- Threshold Analysis ---")
    below_threshold = sum(1 for mag in gyro_magnitudes if mag < threshold)
    percentage_below = (below_threshold / len(gyro_magnitudes)) * 100
    
    print(f"Samples below {threshold} rad/s: {below_threshold}/{len(gyro_magnitudes)} ({percentage_below:.2f}%)")
    
    # Try different thresholds
    print("\n--- Testing Different Thresholds ---")
    test_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    for test_threshold in test_thresholds:
        below = sum(1 for mag in gyro_magnitudes if mag < test_threshold)
        percent = (below / len(gyro_magnitudes)) * 100
        print(f"Threshold {test_threshold:.2f} rad/s: {below}/{len(gyro_magnitudes)} samples ({percent:.2f}%)")
    
    # Check for periods of low activity
    print("\n--- Checking for Static Periods ---")
    window_size = 100  # samples
    static_windows = []
    
    for i in range(len(gyro_magnitudes) - window_size):
        window_max = max(gyro_magnitudes[i:i+window_size])
        if window_max < threshold:
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
        
        # Show some percentiles to understand the distribution
        print("\n--- Percentile Analysis ---")
        sorted_mags = sorted(gyro_magnitudes)
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            index = int(len(sorted_mags) * p / 100)
            value = sorted_mags[min(index, len(sorted_mags)-1)]
            print(f"{p}th percentile: {value:.6f} rad/s")


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
        print("\nLet's check if there are other static experiment files...")
        
        # Try to find the file in other locations
        alt_paths = [
            "../../all_expts/afternoon/Experiments/010_Waiting_for_static_turns/IMU/Sensor_3/gyro_010_Waiting_for_static_turns.csv",
            "../../all_expts/afternoon/Experiments/011_Static_stbd_1/IMU/Sensor_3/gyro_011_Static_stbd_1.csv",
            "../../all_expts/afternoon/Experiments/012_Static_port_1/IMU/Sensor_3/gyro_012_Static_port_1.csv"
        ]
        
        for alt_path in alt_paths:
            full_path = os.path.join(os.path.dirname(__file__), alt_path)
            if os.path.exists(full_path):
                print(f"\nFound alternative file: {alt_path}")
                analyze_gyro_data(full_path)
                return
        
        print("\nCould not find any static experiment gyro files.")
        return
    
    # Analyze the data
    analyze_gyro_data(gyro_file)


if __name__ == "__main__":
    main()