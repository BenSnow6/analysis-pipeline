#!/usr/bin/env python3
"""
Check Sensor_3 orientation in detail.
"""

import numpy as np
import sys
from pathlib import Path

from src.analysis.orientation.orientation_check import OrientationChecker
from src.scripts.frame_definitions import _create_R_bs_from_directions

def check_sensor3():
    """Check Sensor_3 orientation possibilities."""
    
    print("Checking Sensor_3 orientation...")
    print("=" * 60)
    
    # Initialize the orientation checker
    checker = OrientationChecker()
    
    # Load data
    exp_name = "011_Static_stbd_1"
    data = checker.load_aligned_data(exp_name)
    
    sensor_name = "Sensor_3"
    if sensor_name in data and 'accel' in data[sensor_name]:
        accel_data = data[sensor_name]['accel']
        
        # Take static portion
        static_accel = accel_data[:1000]
        mean_accel = np.mean(static_accel, axis=0)
        magnitude = np.linalg.norm(mean_accel)
        gravity_sensor = mean_accel / magnitude
        
        print(f"Gravity in sensor frame: {gravity_sensor}")
        print(f"Magnitude: {magnitude:.1f} m/s²")
        
        # Current configuration says X: Downward, Y: Forward, Z: Port
        # But gravity appears on +X axis, so X is actually pointing down (correct)
        # The 90° error suggests Y or Z mapping is wrong
        
        print("\nTesting different axis configurations:")
        
        # Test different possibilities
        configs = [
            ("Downward", "Forward", "Port"),      # Current
            ("Downward", "Port", "Aft"),          # Y and Z swapped
            ("Downward", "Aft", "Starboard"),     # Different Y and Z
            ("Downward", "Starboard", "Forward"), # Another possibility
        ]
        
        for x_dir, y_dir, z_dir in configs:
            R_bs = _create_R_bs_from_directions(x_dir, y_dir, z_dir)
            
            # Transform gravity to body frame
            gravity_body = R_bs.T @ gravity_sensor
            
            # Expected gravity in body frame
            expected = np.array([0, 0, 1])
            
            # Calculate error
            error = np.arccos(np.clip(np.dot(gravity_body, expected), -1, 1)) * 180 / np.pi
            
            print(f"\nConfig: X={x_dir}, Y={y_dir}, Z={z_dir}")
            print(f"  R_bs:\n{R_bs}")
            print(f"  Gravity in body: {gravity_body}")
            print(f"  Error: {error:.1f}°")
            
            if error < 5:
                print(f"  ✓ This configuration works!")

if __name__ == "__main__":
    check_sensor3()