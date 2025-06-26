#!/usr/bin/env python3
"""
Deduce the actual sensor orientation from gravity measurements.
"""

import numpy as np
import sys
from pathlib import Path

from src.analysis.orientation.orientation_check import OrientationChecker

def deduce_orientation():
    """Deduce actual sensor orientations from gravity measurements."""
    
    print("Deducing actual sensor orientations from gravity measurements...")
    print("=" * 60)
    
    # Initialize the orientation checker
    checker = OrientationChecker()
    
    # Load data for a static experiment
    exp_name = "011_Static_stbd_1"
    data = checker.load_aligned_data(exp_name)
    
    # Check all sensors
    for sensor_name in ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']:
        if sensor_name not in data or 'accel' not in data[sensor_name]:
            continue
            
        accel_data = data[sensor_name]['accel']
        
        # Take a static portion (first 1000 samples)
        static_accel = accel_data[:min(1000, len(accel_data))]
        
        # Calculate mean acceleration (should be gravity)
        mean_accel = np.mean(static_accel, axis=0)
        magnitude = np.linalg.norm(mean_accel)
        
        # Normalize to get direction
        gravity_sensor = mean_accel / magnitude
        
        print(f"\n{sensor_name}:")
        print(f"  Gravity vector: [{gravity_sensor[0]:.3f}, {gravity_sensor[1]:.3f}, {gravity_sensor[2]:.3f}]")
        print(f"  Magnitude: {magnitude:.1f} m/s²")
        
        # Determine which axis gravity aligns with
        abs_gravity = np.abs(gravity_sensor)
        max_axis = np.argmax(abs_gravity)
        axis_names = ['X', 'Y', 'Z']
        
        print(f"  Gravity primarily on {axis_names[max_axis]}-axis ({gravity_sensor[max_axis]:.3f})")
        
        # Deduce orientation
        if max_axis == 0:  # X-axis
            if gravity_sensor[0] > 0:
                print(f"  → Sensor X-axis points DOWNWARD (towards ground)")
            else:
                print(f"  → Sensor X-axis points UPWARD (towards sky)")
        elif max_axis == 1:  # Y-axis
            if gravity_sensor[1] > 0:
                print(f"  → Sensor Y-axis points DOWNWARD (towards ground)")
            else:
                print(f"  → Sensor Y-axis points UPWARD (towards sky)")
        elif max_axis == 2:  # Z-axis
            if gravity_sensor[2] > 0:
                print(f"  → Sensor Z-axis points DOWNWARD (towards ground)")
            else:
                print(f"  → Sensor Z-axis points UPWARD (towards sky)")
                
        # Compare with expected configuration
        sensor_config = checker.config['sensors'][sensor_name]
        print(f"\n  Expected configuration:")
        print(f"    X: {sensor_config['expected_axes']['x_direction']}")
        print(f"    Y: {sensor_config['expected_axes']['y_direction']}")
        print(f"    Z: {sensor_config['expected_axes']['z_direction']}")
        
        # Create the actual rotation matrix based on measurements
        print(f"\n  Creating rotation matrix based on actual measurements...")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    deduce_orientation()