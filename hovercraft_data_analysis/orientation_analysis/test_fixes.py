#!/usr/bin/env python3
"""
Test script to verify orientation analysis fixes.
"""

import sys
from pathlib import Path
from orientation_check import OrientationChecker

def test_single_experiment():
    """Test orientation analysis on a single experiment."""
    
    # Initialize the checker
    checker = OrientationChecker()
    
    # Test with experiment 007
    experiment_name = "007_Fast_stbd_turn_1"
    print(f"\nTesting with experiment: {experiment_name}")
    
    try:
        # Load data
        data = checker.load_aligned_data(experiment_name)
        print(f"Loaded data for sensors: {list(data.keys())}")
        
        # Check if we have timestamp
        if 'timestamp' not in data:
            print("ERROR: No timestamp found in data")
            return
            
        timestamp = data['timestamp']
        print(f"Timestamp length: {len(timestamp)}")
        
        # Test one sensor
        sensor_name = "Sensor_3"
        if sensor_name in data:
            sensor_data = data[sensor_name]
            print(f"\n{sensor_name} data:")
            if 'accel' in sensor_data:
                print(f"  - Accelerometer data shape: {sensor_data['accel'].shape}")
            else:
                print("  - No accelerometer data")
                
            if 'gyro' in sensor_data:
                print(f"  - Gyroscope data shape: {sensor_data['gyro'].shape}")
            else:
                print("  - No gyroscope data - attempting to load from original files")
                
            # Try validation
            if 'accel' in sensor_data and 'gyro' in sensor_data:
                print(f"\nValidating {sensor_name}...")
                results = checker.validate_sensor(
                    sensor_name, 
                    sensor_data, 
                    timestamp, 
                    experiment_name
                )
                
                if 'error' in results:
                    print(f"ERROR: {results['error']}")
                else:
                    print(f"Validation completed!")
                    if 'rotation_error_deg' in results:
                        print(f"Rotation error: {results['rotation_error_deg']:.2f}°")
                        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_experiment()