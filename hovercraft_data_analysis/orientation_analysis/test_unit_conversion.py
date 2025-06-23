#!/usr/bin/env python3
"""
Test the unit conversion fix for accelerometer data.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from orientation_check import OrientationChecker

def test_unit_conversion():
    """Test that accelerometer data is now in m/s² after conversion."""
    
    print("Testing unit conversion fix...")
    print("=" * 60)
    
    # Initialize the orientation checker
    checker = OrientationChecker()
    
    # Try to load a static experiment
    experiments = ["002_Setup", "011_Static_stbd_1", "012_Static_port_1"]
    
    for exp_name in experiments:
        print(f"\nTesting experiment: {exp_name}")
        try:
            # Load the aligned data
            data = checker.load_aligned_data(exp_name)
            
            # Check if we have sensor data
            for sensor_name in ['Sensor_3', 'Sensor_4', 'Sensor_5']:
                if sensor_name in data and 'accel' in data[sensor_name]:
                    accel_data = data[sensor_name]['accel']
                    
                    # Take first 1000 samples (should be static)
                    static_data = accel_data[:min(1000, len(accel_data))]
                    
                    # Calculate statistics
                    mean_accel = np.mean(static_data, axis=0)
                    magnitude = np.linalg.norm(mean_accel)
                    
                    print(f"\n  {sensor_name}:")
                    print(f"    Mean acceleration: [{mean_accel[0]:.3f}, {mean_accel[1]:.3f}, {mean_accel[2]:.3f}] m/s²")
                    print(f"    Magnitude: {magnitude:.3f} m/s²")
                    print(f"    Expected gravity: 9.807 m/s²")
                    print(f"    Error: {abs(magnitude - 9.80665):.3f} m/s² ({abs(magnitude - 9.80665)/9.80665*100:.1f}%)")
                    
                    # Check if conversion worked
                    if 8.0 < magnitude < 11.0:
                        print(f"    ✓ Unit conversion successful!")
                    else:
                        print(f"    ✗ Unit conversion may have failed")
                        
            break  # Just test one experiment that works
            
        except Exception as e:
            print(f"  Could not load {exp_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Unit conversion test complete!")

if __name__ == "__main__":
    test_unit_conversion()