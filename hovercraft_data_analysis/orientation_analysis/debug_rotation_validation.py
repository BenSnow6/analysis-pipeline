#!/usr/bin/env python3
"""
Debug the rotation validation to see what's causing the 180° errors.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from orientation_check import OrientationChecker
from frame_definitions import get_R_bs_dcm

def debug_rotation_validation():
    """Debug the rotation validation process."""
    
    print("Debugging rotation validation...")
    print("=" * 60)
    
    # Initialize the orientation checker
    checker = OrientationChecker()
    
    # Load data for a static experiment
    exp_name = "011_Static_stbd_1"
    data = checker.load_aligned_data(exp_name)
    
    # Focus on Sensor_3
    sensor_name = "Sensor_3"
    if sensor_name in data and 'accel' in data[sensor_name]:
        accel_data = data[sensor_name]['accel']
        gyro_data = data[sensor_name]['gyro']
        timestamp = data['timestamp']
        
        # Take a static portion (first 1000 samples)
        static_accel = accel_data[:1000]
        
        # Calculate mean acceleration (should be gravity)
        mean_accel = np.mean(static_accel, axis=0)
        magnitude = np.linalg.norm(mean_accel)
        
        print(f"\n{sensor_name} Static Analysis:")
        print(f"  Mean acceleration: {mean_accel} m/s²")
        print(f"  Magnitude: {magnitude:.3f} m/s²")
        
        # Normalize to get direction
        gravity_sensor = mean_accel / magnitude
        print(f"  Gravity direction in sensor: {gravity_sensor}")
        
        # Get the rotation matrix
        R_bs = get_R_bs_dcm(sensor_name)
        print(f"\n  Rotation matrix R_bs:")
        print(f"  {R_bs}")
        
        # Transform gravity to body frame
        # The issue might be here - are we transforming correctly?
        print(f"\n  Testing different transformation interpretations:")
        
        # Method 1: R_bs transforms from body to sensor
        gravity_body_1 = R_bs.T @ gravity_sensor
        print(f"    Method 1 (R_bs.T @ g_sensor): {gravity_body_1}")
        
        # Method 2: R_bs transforms from sensor to body  
        gravity_body_2 = R_bs @ gravity_sensor
        print(f"    Method 2 (R_bs @ g_sensor): {gravity_body_2}")
        
        # Expected gravity in body frame (should be [0, 0, 1] for down)
        expected_gravity = np.array([0, 0, 1])
        print(f"\n  Expected gravity in body frame: {expected_gravity}")
        
        # Calculate angles
        angle_1 = np.arccos(np.clip(np.dot(gravity_body_1, expected_gravity), -1, 1)) * 180 / np.pi
        angle_2 = np.arccos(np.clip(np.dot(gravity_body_2, expected_gravity), -1, 1)) * 180 / np.pi
        
        print(f"\n  Angle errors:")
        print(f"    Method 1: {angle_1:.1f}°")
        print(f"    Method 2: {angle_2:.1f}°")
        
        # Check what the rotation validator is doing
        print(f"\n  Running actual rotation validator...")
        rotation_results = checker.rotation_validator.validate_sensor_orientation(
            sensor_name, accel_data, gyro_data, timestamp
        )
        
        print(f"\n  Rotation validation results:")
        print(f"    Gravity in sensor frame: {rotation_results.get('gravity_sensor', 'N/A')}")
        print(f"    Error (current matrix): {rotation_results.get('error_current_deg', 'N/A')}°")
        print(f"    Error (config matrix): {rotation_results.get('error_config_deg', 'N/A')}°")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    debug_rotation_validation()