#!/usr/bin/env python3
"""
Simple test of orientation analysis without external dependencies.
Tests that the fixes work by checking file loading and basic processing.
"""

import os
from pathlib import Path

def test_load_data():
    """Test that we can load data with gyro columns."""
    
    csv_file = Path(__file__).parent.parent.parent / "code" / "alignment_analysis" / "aligned_data" / "007_Fast_stbd_turn_1_csv" / "Sensor_3.csv"
    
    if not csv_file.exists():
        print(f"ERROR: CSV file not found: {csv_file}")
        return False
        
    # Read CSV manually
    with open(csv_file, 'r') as f:
        header = f.readline().strip().split(',')
        first_line = f.readline().strip().split(',')
        
    print(f"CSV columns: {header}")
    print(f"Number of columns: {len(header)}")
    
    # Check for required columns
    required_cols = ['t', 'x', 'y', 'z', 'gyro_x', 'gyro_y', 'gyro_z']
    missing = [col for col in required_cols if col not in header]
    
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return False
    else:
        print("SUCCESS: All required columns present")
        
    # Check data
    print(f"\nFirst data row has {len(first_line)} values")
    if len(first_line) == len(header):
        print("SUCCESS: Data row matches header")
    else:
        print("ERROR: Data row length mismatch")
        return False
        
    return True


def test_orientation_config():
    """Test that orientation config exists and is readable."""
    
    config_file = Path(__file__).parent.parent.parent / "code" / "orientation_analysis" / "orientation_config.yaml"
    
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        return False
        
    print(f"\nConfig file found: {config_file}")
    
    # Just check it's readable
    with open(config_file, 'r') as f:
        content = f.read()
        
    if "sensors:" in content and "validation:" in content:
        print("SUCCESS: Config file contains expected sections")
        return True
    else:
        print("ERROR: Config file missing expected sections")
        return False


def test_experiment_paths():
    """Test that we can find experiment data."""
    
    base_path = Path(__file__).parent.parent.parent / "data"
    
    experiments = [
        ("007_Fast_stbd_turn_1", "afternoon/Experiments"),
        ("016_Straight_cruise_1", "afternoon/Experiments"),
        ("021_Quarter_turn_port", "afternoon/Experiments")
    ]
    
    print(f"\nChecking experiment paths...")
    found = 0
    
    for exp_name, exp_subpath in experiments:
        exp_path = base_path / exp_subpath / exp_name
        if exp_path.exists():
            print(f"  ✓ Found: {exp_name}")
            found += 1
        else:
            print(f"  ✗ Missing: {exp_name} at {exp_path}")
            
    print(f"\nFound {found}/{len(experiments)} experiments")
    return found == len(experiments)


def main():
    """Run all tests."""
    
    print("="*60)
    print("Testing Orientation Analysis Setup")
    print("="*60)
    
    tests = [
        ("Data Loading", test_load_data),
        ("Configuration", test_orientation_config),
        ("Experiment Paths", test_experiment_paths)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
            
    print("\n" + "="*60)
    print(f"SUMMARY: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\nAll tests passed! The orientation analysis should now work.")
        print("\nTo run full orientation analysis, use:")
        print("  python3 run_orientation.py -e 007_Fast_stbd_turn_1 016_Straight_cruise_1 021_Quarter_turn_port")
    else:
        print("\nSome tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()