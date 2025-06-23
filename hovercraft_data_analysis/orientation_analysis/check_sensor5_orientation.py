#!/usr/bin/env python3
"""
Check Sensor_5 orientation in detail.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from orientation_check import OrientationChecker
from frame_definitions import _create_R_bs_from_directions

def check_sensor5():
    """Check Sensor_5 orientation possibilities."""
    
    print("Checking Sensor_5 orientation...")
    print("=" * 60)
    
    # Initialize the orientation checker
    checker = OrientationChecker()
    
    # Load data
    exp_name = "011_Static_stbd_1"
    data = checker.load_aligned_data(exp_name)
    
    sensor_name = "Sensor_5"
    if sensor_name in data and 'accel' in data[sensor_name]:
        accel_data = data[sensor_name]['accel']
        
        # Take static portion
        static_accel = accel_data[:1000]
        mean_accel = np.mean(static_accel, axis=0)
        magnitude = np.linalg.norm(mean_accel)
        gravity_sensor = mean_accel / magnitude
        
        print(f"Gravity in sensor frame: {gravity_sensor}")
        print(f"Components: X={gravity_sensor[0]:.3f}, Y={gravity_sensor[1]:.3f}, Z={gravity_sensor[2]:.3f}")
        print(f"Magnitude: {magnitude:.1f} m/s²")
        
        # Current configuration says X: Forward, Y: Port, Z: Downward
        # Gravity appears as [-0.635, 0.211, 0.744]
        # Z has largest component (0.744) which is correct for Downward
        # But X and Y have significant components too
        
        print("\nTesting different axis configurations:")
        
        # Since gravity has components in all axes, the sensor might be tilted
        # Let's test different orientations
        configs = [
            ("Forward", "Port", "Downward"),      # Current
            ("Aft", "Port", "Downward"),          # X reversed
            ("Forward", "Starboard", "Downward"), # Y reversed
            ("Aft", "Starboard", "Downward"),     # Both reversed
            ("Port", "Aft", "Downward"),          # X/Y swapped
            ("Starboard", "Forward", "Downward"), # Different swap
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
            print(f"  Gravity in body: [{gravity_body[0]:.3f}, {gravity_body[1]:.3f}, {gravity_body[2]:.3f}]")
            print(f"  Error: {error:.1f}°")
            
            if error < 5:
                print(f"  ✓ This configuration works!")
        
        # The 40° error might indicate the sensor is tilted
        print("\n\nNote: The 40° error may indicate the sensor is physically tilted")
        print("relative to the hovercraft body frame, which is expected for a")
        print("steering wheel mounted sensor.")

if __name__ == "__main__":
    check_sensor5()