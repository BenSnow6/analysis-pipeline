#!/usr/bin/env python3
"""
Check if gyroscope data is in degrees/s instead of rad/s
"""

import csv
import math
import sys
import os


def check_units(gyro_file):
    """Check if gyro data might be in degrees instead of radians."""
    
    print(f"\nChecking units for: {gyro_file}")
    print("-" * 80)
    
    # Load the data
    with open(gyro_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = list(reader)
    
    # Find gyro columns
    gyro_indices = []
    for i, col in enumerate(headers):
        if 'x' in col.lower() or 'y' in col.lower() or 'z' in col.lower():
            gyro_indices.append(i)
    
    if len(gyro_indices) < 3:
        print("Could not find x, y, z columns")
        return
    
    # Get first 1000 samples
    magnitudes_raw = []
    magnitudes_deg_to_rad = []
    
    for row in data[:1000]:
        try:
            gx = float(row[gyro_indices[0]])
            gy = float(row[gyro_indices[1]])
            gz = float(row[gyro_indices[2]])
            
            # Raw magnitude (assuming already in rad/s)
            mag_raw = math.sqrt(gx**2 + gy**2 + gz**2)
            magnitudes_raw.append(mag_raw)
            
            # Convert from deg/s to rad/s
            gx_rad = gx * (math.pi / 180.0)
            gy_rad = gy * (math.pi / 180.0)
            gz_rad = gz * (math.pi / 180.0)
            mag_converted = math.sqrt(gx_rad**2 + gy_rad**2 + gz_rad**2)
            magnitudes_deg_to_rad.append(mag_converted)
            
        except (ValueError, IndexError):
            continue
    
    # Calculate statistics
    mean_raw = sum(magnitudes_raw) / len(magnitudes_raw)
    mean_converted = sum(magnitudes_deg_to_rad) / len(magnitudes_deg_to_rad)
    
    print(f"\nIf data is in rad/s (as is):")
    print(f"  Mean magnitude: {mean_raw:.6f} rad/s")
    print(f"  Min magnitude: {min(magnitudes_raw):.6f} rad/s")
    print(f"  Max magnitude: {max(magnitudes_raw):.6f} rad/s")
    print(f"  This corresponds to {mean_raw * 180 / math.pi:.1f} deg/s average")
    
    print(f"\nIf data is in deg/s (converted to rad/s):")
    print(f"  Mean magnitude: {mean_converted:.6f} rad/s")
    print(f"  Min magnitude: {min(magnitudes_deg_to_rad):.6f} rad/s")
    print(f"  Max magnitude: {max(magnitudes_deg_to_rad):.6f} rad/s")
    print(f"  This corresponds to {mean_converted * 180 / math.pi:.1f} deg/s average")
    
    # Check how many would be below threshold
    threshold = 0.05  # rad/s
    below_raw = sum(1 for m in magnitudes_raw if m < threshold)
    below_converted = sum(1 for m in magnitudes_deg_to_rad if m < threshold)
    
    print(f"\nSamples below {threshold} rad/s threshold:")
    print(f"  If already in rad/s: {below_raw}/{len(magnitudes_raw)} ({100*below_raw/len(magnitudes_raw):.1f}%)")
    print(f"  If converted from deg/s: {below_converted}/{len(magnitudes_deg_to_rad)} ({100*below_converted/len(magnitudes_deg_to_rad):.1f}%)")
    
    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    if mean_converted < 0.5 and below_converted > len(magnitudes_deg_to_rad) * 0.8:
        print("The data appears to be in DEGREES/SECOND!")
        print("You should convert to rad/s by multiplying by π/180")
        print(f"After conversion, mean would be {mean_converted:.3f} rad/s")
    else:
        print("The data might already be in rad/s, but values seem too high for static data")
        print("This could indicate:")
        print("1. Sensor noise or calibration issues")
        print("2. The vehicle was not actually static")
        print("3. Incorrect sensor configuration")


def main():
    # Import path configuration
    from pathlib import Path
    from src.config.paths import get_experiment_path
    
    if len(sys.argv) > 1:
        gyro_file = sys.argv[1]
    else:
        # Use centralized path helper
        exp_path = get_experiment_path("010_Waiting_for_static_turns", "afternoon")
        gyro_file = str(exp_path / "IMU" / "Sensor_3" / "gyro_010_Waiting_for_static_turns.csv")
    
    if os.path.exists(gyro_file):
        check_units(gyro_file)
        
        # Also check other static experiments
        print("\n" + "="*80)
        print("CHECKING OTHER STATIC EXPERIMENTS")
        print("="*80)
        
        other_experiments = [
            ("011_Static_stbd_1", "gyro_011_Static_stbd_1.csv"),
            ("012_Static_port_1", "gyro_012_Static_port_1.csv")
        ]
        
        for exp_name, gyro_filename in other_experiments:
            exp_path = get_experiment_path(exp_name, "afternoon")
            full_path = exp_path / "IMU" / "Sensor_3" / gyro_filename
            if full_path.exists():
                check_units(str(full_path))
    else:
        print(f"File not found: {gyro_file}")


if __name__ == "__main__":
    main()