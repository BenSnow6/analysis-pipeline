#!/usr/bin/env python3
"""Debug script to check data loading and static detection."""

import pandas as pd
import numpy as np
from pathlib import Path
from static_detector import StaticDetector
from orientation_check import OrientationChecker

# Check the aligned data
aligned_dir = Path(__file__).parent.parent / "alignment_analysis" / "aligned_data" / "007_Fast_stbd_turn_1_csv"
sensor_file = aligned_dir / "Sensor_3.csv"

print(f"Loading {sensor_file}")
df = pd.read_csv(sensor_file)

print(f"\nColumns: {list(df.columns)}")
print(f"Shape: {df.shape}")

# Check accelerometer data
print("\nAccelerometer data:")
accel_data = df[['x', 'y', 'z']].values
print(f"  Shape: {accel_data.shape}")
print(f"  NaN count: {np.sum(np.isnan(accel_data))}")
print(f"  First 5 rows:\n{accel_data[:5]}")

# Check gyroscope data
print("\nGyroscope data:")
if 'gyro_x' in df.columns:
    gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
    print(f"  Shape: {gyro_data.shape}")
    print(f"  NaN count: {np.sum(np.isnan(gyro_data))}")
    print(f"  Non-NaN count: {np.sum(~np.isnan(gyro_data[:, 0]))}")
    
    # Find first non-NaN indices
    non_nan_mask = ~np.isnan(gyro_data[:, 0])
    non_nan_indices = np.where(non_nan_mask)[0]
    if len(non_nan_indices) > 0:
        print(f"  First non-NaN index: {non_nan_indices[0]}")
        print(f"  Last non-NaN index: {non_nan_indices[-1]}")
        print(f"  First 5 non-NaN rows:")
        print(gyro_data[non_nan_indices[:5]])
else:
    print("  ERROR: gyro columns not found!")

# Check timestamps
print("\nTimestamps:")
timestamps = df['t'].values
print(f"  Shape: {timestamps.shape}")
print(f"  Range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} seconds")
print(f"  Sample rate: {1.0 / np.mean(np.diff(timestamps)):.2f} Hz")

# Test static detection
print("\n\nTesting static detection...")
static_detector = StaticDetector()

# Try to detect static segments
try:
    static_segments = static_detector.detect_static_segments(
        timestamps, gyro_data, accel_data
    )
    print(f"Found {len(static_segments)} static segments:")
    for i, (start, end) in enumerate(static_segments):
        duration = end - start
        print(f"  Segment {i+1}: {start:.2f}s to {end:.2f}s (duration: {duration:.2f}s)")
except Exception as e:
    print(f"ERROR in static detection: {e}")
    import traceback
    traceback.print_exc()

# Test with only valid data
print("\n\nTesting with only valid (non-NaN) data...")
valid_mask = ~np.isnan(gyro_data[:, 0])
valid_timestamps = timestamps[valid_mask]
valid_gyro = gyro_data[valid_mask]
valid_accel = accel_data[valid_mask]

print(f"Valid data shape: {valid_gyro.shape}")
print(f"Valid time range: {valid_timestamps[0]:.2f} to {valid_timestamps[-1]:.2f} seconds")

try:
    static_segments = static_detector.detect_static_segments(
        valid_timestamps, valid_gyro, valid_accel
    )
    print(f"Found {len(static_segments)} static segments in valid data:")
    for i, (start, end) in enumerate(static_segments):
        duration = end - start
        print(f"  Segment {i+1}: {start:.2f}s to {end:.2f}s (duration: {duration:.2f}s)")
except Exception as e:
    print(f"ERROR in static detection: {e}")
    import traceback
    traceback.print_exc()