#!/usr/bin/env python3
"""
Add gyroscope data to static experiment aligned CSV files.
This script processes experiments in the aligned_data/static directories.
"""

import csv
from pathlib import Path


def add_gyro_data(experiment_name, csv_dir, experiment_data_paths):
    """Add gyro data to existing CSV files for an experiment."""
    
    print(f"\nProcessing {experiment_name}...")
    
    # Process each sensor
    for sensor_name in ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']:
        sensor_csv = csv_dir / f"{sensor_name}.csv"
        
        if not sensor_csv.exists():
            print(f"  Skipping {sensor_name} - no CSV file")
            continue
            
        # Find gyro data file - try multiple possible locations
        gyro_file = None
        for exp_data_path in experiment_data_paths:
            possible_gyro_paths = [
                exp_data_path / "IMU" / sensor_name / f"gyro_{experiment_name}.csv",
                exp_data_path / sensor_name / f"gyro_{experiment_name}.csv"
            ]
            
            for path in possible_gyro_paths:
                if path.exists():
                    gyro_file = path
                    break
            if gyro_file:
                break
                
        if not gyro_file:
            print(f"  WARNING: No gyro data found for {sensor_name}")
            continue
            
        # Read existing CSV data
        with open(sensor_csv, 'r') as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
            
        if not existing_data:
            print(f"  ERROR: No data in {sensor_csv}")
            continue
            
        # Read gyro data
        with open(gyro_file, 'r') as f:
            reader = csv.DictReader(f)
            gyro_data = list(reader)
            
        # Check if we already have gyro data
        if 'gyro_x' in existing_data[0]:
            print(f"  {sensor_name} already has gyro data")
            continue
            
        # Match gyro data to existing data by index (assuming same length)
        if len(gyro_data) != len(existing_data):
            print(f"  WARNING: Length mismatch for {sensor_name} - accel: {len(existing_data)}, gyro: {len(gyro_data)}")
            # Use minimum length
            min_len = min(len(existing_data), len(gyro_data))
            existing_data = existing_data[:min_len]
            gyro_data = gyro_data[:min_len]
            
        # Add gyro columns to existing data
        for i, (row, gyro_row) in enumerate(zip(existing_data, gyro_data)):
            row['gyro_x'] = gyro_row['x']
            row['gyro_y'] = gyro_row['y']
            row['gyro_z'] = gyro_row['z']
            
        # Write updated data back
        # Get all fieldnames from the first row
        if existing_data:
            fieldnames = list(existing_data[0].keys())
            # Ensure gyro fields are at the end if not already present
            for field in ['gyro_x', 'gyro_y', 'gyro_z']:
                if field not in fieldnames:
                    fieldnames.append(field)
        else:
            fieldnames = ['t', 'x', 'y', 'z', 'time_from_sync', 'gyro_x', 'gyro_y', 'gyro_z']
            
        with open(sensor_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
            
        print(f"  Added gyro data to {sensor_name}")
        
    print(f"Completed {experiment_name}")
    return True


def main():
    """Process static experiments to add gyro data."""
    
    # Base paths
    aligned_static_path = Path(__file__).parent.parent / "alignment_analysis" / "aligned_data" / "static"
    all_expts_path = Path(__file__).parent.parent.parent / "all_expts"
    
    # Static experiments to process
    static_experiments = [
        # Afternoon experiments
        ("002_Setup", "afternoon"),
        ("003_Waiting_for_departure", "afternoon"),
        ("010_Waiting_for_static_turns", "afternoon"),
        # Morning experiments
        ("002_Setup", "morning"),
        ("004_Setup_2", "morning")
    ]
    
    success_count = 0
    
    for exp_name, time_period in static_experiments:
        # CSV directory in aligned_data/static
        csv_dir = aligned_static_path / time_period / f"{exp_name}_csv"
        
        if not csv_dir.exists():
            print(f"No CSV directory found for {exp_name} ({time_period})")
            continue
            
        # Possible locations for raw data
        experiment_data_paths = [
            all_expts_path / time_period / "Experiments" / exp_name,
            all_expts_path / time_period / exp_name
        ]
        
        # Check if at least one path exists
        valid_paths = [p for p in experiment_data_paths if p.exists()]
        if not valid_paths:
            print(f"ERROR: No experiment data found for {exp_name} ({time_period})")
            continue
            
        if add_gyro_data(exp_name, csv_dir, valid_paths):
            success_count += 1
            
    print(f"\n{'='*60}")
    print(f"Processed {success_count}/{len(static_experiments)} experiments successfully")
    
    # Also check for any other CSV directories that might exist
    print("\n\nChecking for other CSV directories in static folders...")
    for time_period in ["morning", "afternoon"]:
        time_dir = aligned_static_path / time_period
        if time_dir.exists():
            csv_dirs = list(time_dir.glob("*_csv"))
            for csv_dir in csv_dirs:
                exp_name = csv_dir.name.replace("_csv", "")
                if not any(exp[0] == exp_name and exp[1] == time_period for exp in static_experiments):
                    print(f"  Found additional experiment: {exp_name} ({time_period})")


if __name__ == "__main__":
    main()