#!/usr/bin/env python3
"""
Simple plotting script for alignment analysis.
Uses only matplotlib and csv - no pandas required.
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path


def read_csv_data(filepath):
    """Read CSV file and return lists of data."""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        # Initialize lists for each column
        for row in reader:
            if not data:  # First row - initialize
                for key in row.keys():
                    data[key] = []
            # Add values
            for key, value in row.items():
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(value)
            
    return data


def plot_experiment(experiment_name):
    """Create plots for alignment analysis."""
    csv_dir = Path(f'aligned_data/{experiment_name}_csv')
    
    if not csv_dir.exists():
        print(f"Error: Directory {csv_dir} not found!")
        return
    
    print(f"\nPlotting {experiment_name}...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Sensors to plot
    sensors = ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']
    plot_idx = 0
    
    for sensor in sensors:
        csv_file = csv_dir / f'{sensor}.csv'
        if not csv_file.exists():
            continue
            
        # Read data
        data = read_csv_data(csv_file)
        
        if 'time_diff_ms' in data and data['time_diff_ms']:
            ax = axes[plot_idx]
            
            # Create histogram
            time_diffs = [d for d in data['time_diff_ms'] if d is not None]
            
            ax.hist(time_diffs, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Time Difference (ms)')
            ax.set_ylabel('Count')
            ax.set_title(f'{sensor} Time Alignment')
            
            # Add statistics
            if time_diffs:
                mean_diff = sum(time_diffs) / len(time_diffs)
                max_diff = max(time_diffs)
                text = f'Mean: {mean_diff:.3f} ms\nMax: {max_diff:.3f} ms'
                ax.text(0.95, 0.95, text, transform=ax.transAxes, 
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Alignment Quality: {experiment_name}')
    plt.tight_layout()
    
    # Save figure
    output_file = f'alignment_quality_{experiment_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()
    
    # Create time series plot
    plt.figure(figsize=(14, 6))
    
    for sensor in ['Sensor_4', 'Sensor_5', 'Sensor_wb']:
        csv_file = csv_dir / f'{sensor}.csv'
        if not csv_file.exists():
            continue
            
        data = read_csv_data(csv_file)
        
        if 'aligned_time' in data and 'time_diff_ms' in data:
            times = data['aligned_time']
            diffs = data['time_diff_ms']
            
            # Sample every 100th point to avoid overplotting
            times_sample = times[::100]
            diffs_sample = diffs[::100]
            
            plt.plot(times_sample, diffs_sample, 'o', markersize=2, 
                    alpha=0.6, label=sensor)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Time Difference (ms)')
    plt.title(f'Alignment Consistency: {experiment_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = f'alignment_consistency_{experiment_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def main():
    """Create plots for all experiments."""
    print("Alignment Quality Visualization")
    print("==============================")
    
    aligned_dir = Path('aligned_data')
    csv_dirs = sorted(aligned_dir.glob('*_csv'))
    
    if not csv_dirs:
        print("No CSV directories found!")
        return
    
    # Process each experiment
    for csv_dir in csv_dirs:
        exp_name = csv_dir.name.replace('_csv', '').replace('_aligned', '')
        plot_experiment(exp_name)
    
    print("\nAll plots generated!")


if __name__ == "__main__":
    main()