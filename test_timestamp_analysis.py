#!/usr/bin/env python3
"""
Quick test script for timestamp analysis tool.
Run this from the analysis-pipeline directory.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hovercraft_data_analysis.timestamp_analysis import data_loader, timestamp_analyzer

def test_basic_functionality():
    """Test basic functionality of the timestamp analysis tool."""
    print("Testing timestamp analysis tool...")
    
    # Load sensor specs
    specs = data_loader.load_sensor_specs()
    print(f"✓ Loaded sensor specifications")
    
    # Get available experiments
    experiments = data_loader.get_available_experiments()
    print(f"✓ Found {len(experiments)} experiments")
    
    if experiments:
        # Test with first available experiment
        first_exp = list(experiments.keys())[0]
        print(f"\nTesting with experiment: {first_exp}")
        
        # Load data
        exp_path = experiments[first_exp]
        sensor_data = data_loader.load_experiment_data(exp_path, specs)
        print(f"✓ Loaded data for {len(sensor_data)} sensors")
        
        # Analyze timestamps
        if sensor_data:
            results = timestamp_analyzer.analyze_experiment(sensor_data, specs)
            print(f"✓ Analysis completed for {len(results)} sensors")
            
            # Print summary
            print("\nSummary:")
            for sensor_name, result in results.items():
                status = "PASS" if result.within_spec else "FAIL"
                print(f"  {sensor_name}: {status} "
                      f"(Rate: {result.actual_rate_hz:.1f}Hz, "
                      f"Jitter: {result.mean_jitter_ms:.1f}ms)")
        else:
            print("⚠ No sensor data found for test experiment")
    else:
        print("⚠ No experiments found in data repository")
    
    print("\n✓ Basic functionality test completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()