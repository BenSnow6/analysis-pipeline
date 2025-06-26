#!/usr/bin/env python3
"""
Add gyroscope data to existing aligned CSV files.
This is a simple script that doesn't require pandas or numpy.
"""

import csv
from pathlib import Path


def add_gyro_data(experiment_name, csv_dir, experiment_data_path):
    """Add gyro data to existing CSV files for an experiment."""
    
    print(f"\nProcessing {experiment_name}...")
    
    # Process each sensor
    for sensor_name in ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']:
        sensor_csv = csv_dir / f"{sensor_name}.csv"
        
        if not sensor_csv.exists():
            print(f"  Skipping {sensor_name} - no CSV file")
            continue
            
        # Find gyro data file
        gyro_file = None
        possible_gyro_paths = [
            experiment_data_path / "IMU" / sensor_name / f"gyro_{experiment_name}.csv",
            experiment_data_path / sensor_name / f"gyro_{experiment_name}.csv"
        ]
        
        for path in possible_gyro_paths:
            if path.exists():
                gyro_file = path
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
    """Process key experiments to add gyro data."""
    
    # Base paths
    base_path = Path(__file__).parent.parent.parent / "data/raw"
    aligned_path = Path(__file__).parent.parent / "alignment_analysis" / "aligned_data"
    
    # Key experiments for orientation analysis
    experiments = [
        ("007_Fast_stbd_turn_1", "1a_1_Minimum_Radius_Turn/afternoon"),
        ("016_Straight_cruise_1", "1b_1_Ground_Acceleration_Time_and_Distance/afternoon"),
        ("021_Quarter_turn_port", "1a_2_Rate_of_Turn_vs_Nosewheel_Steering_Angle/afternoon"),
        # Static experiments
        ("011_Static_stbd_1", "1a_1_Minimum_Radius_Turn/afternoon"),
        ("012_Static_port_1", "1a_1_Minimum_Radius_Turn/afternoon"),
        ("013_Static_port_2", "1a_1_Minimum_Radius_Turn/afternoon"),
        ("014_Static_stbd_2", "1a_1_Minimum_Radius_Turn/afternoon")
    ]
    
    success_count = 0
    
    for exp_name, exp_subpath in experiments:
        # Find CSV directory
        csv_dirs = [
            aligned_path / f"{exp_name}_csv",
            aligned_path / "afternoon" / f"{exp_name}_csv"
        ]
        
        csv_dir = None
        for d in csv_dirs:
            if d.exists():
                csv_dir = d
                break
                
        if not csv_dir:
            print(f"ERROR: No CSV directory found for {exp_name}")
            continue
            
        # Find experiment data
        exp_data_path = base_path / exp_subpath / exp_name
        if not exp_data_path.exists():
            print(f"ERROR: No experiment data found at {exp_data_path}")
            continue
            
        if add_gyro_data(exp_name, csv_dir, exp_data_path):
            success_count += 1
            
    print(f"\n{'='*60}")
    print(f"Processed {success_count}/{len(experiments)} experiments successfully")
    

if __name__ == "__main__":
    main()