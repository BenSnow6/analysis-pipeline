#!/usr/bin/env python3
"""Test script to run WP-3 processing"""

from pathlib import Path

# Check what data is available
def check_data():
    base_path = Path(__file__).parent.parent.parent / 'code' / 'alignment_analysis' / 'aligned_data'
    
    print(f"Looking for data in: {base_path}")
    print(f"Path exists: {base_path.exists()}")
    
    if base_path.exists():
        # Check afternoon directory
        afternoon_dir = base_path / 'afternoon'
        print(f"\nAfternoon directory exists: {afternoon_dir.exists()}")
        
        if afternoon_dir.exists():
            print("\nAvailable experiments in afternoon:")
            for exp_dir in sorted(afternoon_dir.glob("*_csv")):
                print(f"  - {exp_dir.name}")
                # Check for sensor files
                sensor_files = list(exp_dir.glob("Sensor_*.csv"))
                if sensor_files:
                    print(f"    Sensors: {', '.join(f.stem for f in sensor_files)}")
    
    # Check WP-1 output
    wp1_path = Path(__file__).parent.parent.parent / 'code' / 'rpm_estimation' / 'results' / 'wp1' / 'output_wp1' / 'afternoon'
    print(f"\nWP-1 output path: {wp1_path}")
    print(f"WP-1 path exists: {wp1_path.exists()}")
    
    if wp1_path.exists():
        print("\nWP-1 processed experiments:")
        for exp_dir in sorted(wp1_path.iterdir()):
            if exp_dir.is_dir():
                print(f"  - {exp_dir.name}")
                parquet_files = list(exp_dir.glob("*.parquet"))
                if parquet_files:
                    print(f"    Files: {', '.join(f.name for f in parquet_files)}")

if __name__ == "__main__":
    check_data()