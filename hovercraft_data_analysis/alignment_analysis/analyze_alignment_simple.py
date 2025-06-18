#!/usr/bin/env python3
"""
Simple alignment analysis script that works with CSV files.
Minimal dependencies - only requires basic Python libraries.
"""

import csv
import os
from pathlib import Path
import statistics


def read_csv_simple(filepath):
    """Read CSV file using built-in csv module."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric values
            for key in row:
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass  # Keep as string
            data.append(row)
    return data


def analyze_experiment(experiment_name):
    """Analyze alignment quality for an experiment using CSV files."""
    csv_dir = Path(f'aligned_data/{experiment_name}_csv')
    
    if not csv_dir.exists():
        print(f"Error: Directory {csv_dir} not found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # List all CSV files
    csv_files = sorted(csv_dir.glob('*.csv'))
    print(f"\nFound {len(csv_files)} files:")
    
    for csv_file in csv_files:
        if csv_file.stem == 'metadata':
            continue
            
        print(f"\n{csv_file.stem}:")
        print("-" * 40)
        
        # Read data
        data = read_csv_simple(csv_file)
        if not data:
            print("  No data")
            continue
            
        # Basic statistics
        print(f"  Samples: {len(data)}")
        
        # Check for time columns
        first_row = data[0]
        time_col = None
        if 'aligned_time' in first_row:
            time_col = 'aligned_time'
        elif 'time_from_sync' in first_row:
            time_col = 'time_from_sync'
        
        if time_col and len(data) > 1:
            # Calculate duration and rate
            times = [row[time_col] for row in data]
            duration = max(times) - min(times)
            rate = len(data) / duration if duration > 0 else 0
            print(f"  Duration: {duration:.1f} seconds")
            print(f"  Sample rate: {rate:.1f} Hz")
            
            # Time differences between samples
            time_diffs = []
            for i in range(1, len(times)):
                time_diffs.append((times[i] - times[i-1]) * 1000)  # Convert to ms
            
            if time_diffs:
                mean_period = statistics.mean(time_diffs)
                print(f"  Mean period: {mean_period:.2f} ms")
        
        # Check for alignment quality metrics
        if 'time_diff_ms' in first_row:
            time_diffs_ms = [row['time_diff_ms'] for row in data if row['time_diff_ms'] is not None]
            if time_diffs_ms:
                mean_diff = statistics.mean(time_diffs_ms)
                max_diff = max(time_diffs_ms)
                min_diff = min(time_diffs_ms)
                print(f"  Alignment quality:")
                print(f"    Mean time diff: {mean_diff:.3f} ms")
                print(f"    Max time diff: {max_diff:.3f} ms")
                print(f"    Min time diff: {min_diff:.3f} ms")
                
                # Simple histogram
                print(f"  Time diff distribution:")
                bins = [0, 1, 2, 3, 4, 5]
                for i in range(len(bins)-1):
                    count = sum(1 for d in time_diffs_ms if bins[i] <= d < bins[i+1])
                    bar = '#' * int(count / len(time_diffs_ms) * 50)
                    print(f"    {bins[i]}-{bins[i+1]}ms: {bar} ({count})")


def main():
    """Analyze all available experiments."""
    print("Hovercraft Data Alignment Analysis")
    print("==================================")
    
    aligned_dir = Path('aligned_data')
    if not aligned_dir.exists():
        print("Error: aligned_data directory not found!")
        return
    
    # Find all CSV directories
    csv_dirs = sorted(aligned_dir.glob('*_csv'))
    
    if not csv_dirs:
        print("No CSV directories found!")
        print("Please run export_to_csv.py first.")
        return
    
    print(f"\nFound {len(csv_dirs)} experiments:")
    experiments = []
    for d in csv_dirs:
        exp_name = d.name.replace('_csv', '').replace('_aligned', '')
        experiments.append(exp_name)
        print(f"  {len(experiments)}. {exp_name}")
    
    # Analyze each experiment
    for exp in experiments:
        analyze_experiment(exp)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    
    # Summary
    print("\nQuick reference:")
    print("- Sensor_3: Reference sensor (200 Hz)")
    print("- Sensor_4, Sensor_5: 200 Hz sensors")
    print("- Sensor_wb: 100 Hz sensor (downsampled 2:1)")
    print("- gps: 1 Hz GPS data")
    print("- Sensor_wnb: Excluded due to timing issues")


if __name__ == "__main__":
    main()