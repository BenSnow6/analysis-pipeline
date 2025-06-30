#!/usr/bin/env python3
"""
Generate plots from raw data files.
This script shows how to recreate the plots that were removed from /data/raw/.
Plots are saved to /visualizations/ instead of polluting the raw data directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import numpy as np

from src.core.paths import RAW_DATA_DIR, PROJECT_ROOT

# Create visualizations directory
VIZ_DIR = PROJECT_ROOT / "visualizations"

def plot_gps_path(experiment_name: str, time_of_day: str, output_dir: Optional[Path] = None):
    """Generate GPS path plot for an experiment."""
    if output_dir is None:
        output_dir = VIZ_DIR / time_of_day / experiment_name / "GPS"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load GPS data
    gps_file = RAW_DATA_DIR / time_of_day / "Experiments" / experiment_name / "GPS" / f"GPS_{experiment_name}.csv"
    if not gps_file.exists():
        print(f"GPS file not found: {gps_file}")
        return
    
    df = pd.read_csv(gps_file)
    
    # Create path plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df['Lng'], df['Lat'], 'b-', linewidth=2)
    ax.scatter(df['Lng'].iloc[0], df['Lat'].iloc[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(df['Lng'].iloc[-1], df['Lat'].iloc[-1], c='red', s=100, marker='s', label='End')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'GPS Path - {experiment_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    output_file = output_dir / f"GPS_Path_{experiment_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

def plot_imu_sensor(experiment_name: str, time_of_day: str, sensor_name: str, 
                   measurement_type: str, output_dir: Optional[Path] = None):
    """Generate IMU sensor plot for an experiment."""
    if output_dir is None:
        output_dir = VIZ_DIR / time_of_day / experiment_name / "IMU" / sensor_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load IMU data
    imu_file = (RAW_DATA_DIR / time_of_day / "Experiments" / experiment_name / 
                "IMU" / sensor_name / f"{measurement_type}_{experiment_name}.csv")
    
    if not imu_file.exists():
        print(f"IMU file not found: {imu_file}")
        return
    
    df = pd.read_csv(imu_file)
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Assuming columns are like 'timestamp', 'x', 'y', 'z' or similar
    time_col = df.columns[0]  # Usually timestamp
    
    labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        if i + 1 < len(df.columns):
            ax.plot(df[time_col], df.iloc[:, i + 1], color=color, linewidth=1)
            ax.set_ylabel(f'{label} axis')
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle(f'{sensor_name} - {measurement_type} - {experiment_name}')
    
    output_file = output_dir / f"{experiment_name}_{sensor_name}_{measurement_type}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

def generate_experiment_plots(experiment_name: str, time_of_day: str = "afternoon"):
    """Generate all plots for a single experiment."""
    print(f"\nGenerating plots for {experiment_name} ({time_of_day})...")
    
    # GPS plot
    plot_gps_path(experiment_name, time_of_day)
    
    # IMU plots
    sensors = ["Sensor_3", "Sensor_4", "Sensor_5", "Sensor_wb", "Sensor_wnb"]
    measurements = ["accel", "gyro", "angle", "mag"]
    
    for sensor in sensors:
        for measurement in measurements:
            plot_imu_sensor(experiment_name, time_of_day, sensor, measurement)

def main():
    """Example usage."""
    print("Data Plot Generator")
    print("=" * 50)
    print(f"Plots will be saved to: {VIZ_DIR}")
    print("\nThis script demonstrates how to regenerate plots from raw data.")
    print("Plots are saved to /visualizations/ to keep /data/raw/ clean.")
    
    # Example: Generate plots for one experiment
    # Uncomment to run:
    # generate_experiment_plots("007_Fast_stbd_turn_1", "afternoon")
    
    print("\nTo generate plots, call generate_experiment_plots() with experiment name.")

if __name__ == "__main__":
    main()