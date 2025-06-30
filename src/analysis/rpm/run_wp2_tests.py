#!/usr/bin/env python3
"""
Run WP2 tests with proper column name mapping for aligned data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from spectral import extract_rpm_from_vibration
from tracking import RPMTimeSeries
import h5py
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from wp2_process import load_config, save_rpm_results, create_diagnostic_plots


def load_and_preprocess_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV data and map column names correctly."""
    logger.info(f"Loading data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Map column names
    column_mapping = {
        'x': 'ax',
        'y': 'ay', 
        'z': 'az',
        't': 'time_from_sync'
    }
    
    # Rename columns if needed
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Apply high-pass filter to remove gravity
    from scipy import signal
    fs = 200  # Hz
    cutoff = 5  # Hz
    
    b, a = signal.butter(4, cutoff / (fs/2), 'high')
    
    # Filter each component
    for col in ['ax', 'ay', 'az']:
        if col in df.columns:
            df[f'{col}_hp'] = signal.filtfilt(b, a, df[col])
    
    # Compute magnitude
    df['a_hp_mag'] = np.sqrt(df['ax_hp']**2 + df['ay_hp']**2 + df['az_hp']**2)
    
    return df


def process_experiment_test(experiment: str, session: str, config: dict,
                           base_path: Path, output_base: Path) -> dict:
    """Process a single experiment with proper column handling."""
    
    results = {}
    sensors = ['Sensor_3', 'Sensor_4', 'Sensor_wb']
    
    # Create output directories
    output_dir = output_base / 'results' / 'wp2' / session
    plot_dir = output_base / 'results' / 'wp2' / 'plots' / session
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for sensor_id in sensors:
        logger.info(f"Processing {experiment}/{sensor_id} ({session})")
        
        # Load data - check multiple possible locations
        possible_paths = [
            base_path / session / f"{experiment}_csv" / f"{sensor_id}.csv",
            base_path / "static" / session / f"{experiment}_csv" / f"{sensor_id}.csv"
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = path
                break
                
        if csv_path is None:
            logger.warning(f"Data not found for {experiment}/{sensor_id}")
            continue
            
        # Load and preprocess
        data = load_and_preprocess_data(csv_path)
        
        # Process in windows
        window_seconds = 30.0
        hop_seconds = 15.0
        fs = config['fs']
        window_samples = int(window_seconds * fs)
        hop_samples = int(hop_seconds * fs)
        
        vibration_mag = data['a_hp_mag'].values
        times = data['time_from_sync'].values
        
        rpm_frames = []
        
        # Process windows
        for start_idx in range(0, len(vibration_mag) - window_samples + 1, hop_samples):
            end_idx = start_idx + window_samples
            
            window_data = vibration_mag[start_idx:end_idx]
            window_time = times[start_idx + window_samples // 2]
            
            rpm_frame = extract_rpm_from_vibration(
                window_data, fs, config,
                timestamp=window_time,
                sensor_id=sensor_id
            )
            
            if rpm_frame is not None:
                rpm_frames.append(rpm_frame)
        
        if not rpm_frames:
            logger.warning(f"No valid RPM estimates for {experiment}/{sensor_id}")
            continue
            
        # Create RPM series
        rpm_series = RPMTimeSeries(
            experiment=experiment,
            session=session,
            sensor_id=sensor_id,
            frames=rpm_frames
        )
        
        # Save results
        output_path = output_dir / f"{experiment}_{sensor_id}_rpm.h5"
        save_rpm_results(rpm_series, output_path)
        
        # Create diagnostic plot
        create_diagnostic_plots(rpm_series, data, config, plot_dir)
        
        # Print summary
        valid_frames = [f for f in rpm_frames if f.is_valid()]
        if valid_frames:
            mean_rpm = np.mean([f.rpm for f in valid_frames])
            availability = len(valid_frames) / len(rpm_frames) * 100
            logger.info(f"  Mean RPM: {mean_rpm:.0f}")
            logger.info(f"  Availability: {availability:.1f}%")
            
        results[sensor_id] = output_path
        
    return results


def main():
    """Run WP2 tests on key experiments."""
    
    # Load configuration
    config_path = Path(__file__).parent / 'rpm_config.yaml'
    config = load_config(config_path)
    
    # Base paths
    base_path = Path(__file__).parent.parent.parent / 'hovercraft_data_analysis' / 'alignment_analysis' / 'aligned_data'
    output_base = Path(__file__).parent
    
    # Test experiments
    test_experiments = [
        ("007_Fast_stbd_turn_1", "afternoon"),     # Dynamic maneuver
        ("003_Waiting_for_departure", "afternoon"), # Static idle test
        ("026_Engine_rpm_sweep", "afternoon"),      # RPM sweep validation
    ]
    
    print("\nWP-2 Test Processing")
    print("=" * 60)
    
    for exp_name, session in test_experiments:
        print(f"\nProcessing {exp_name} ({session})...")
        
        try:
            results = process_experiment_test(exp_name, session, config, base_path, output_base)
            
            if results:
                print(f"✓ Successfully processed {exp_name}")
                print(f"  Output files: {len(results)}")
            else:
                print(f"✗ No results for {exp_name}")
                
        except Exception as e:
            print(f"✗ Error processing {exp_name}: {e}")
            logger.exception(f"Failed to process {exp_name}")
    
    print("\n" + "=" * 60)
    print("Test processing complete!")
    print("\nCheck results in:")
    print("- results/wp2/afternoon/  (HDF5 files)")
    print("- results/wp2/plots/afternoon/  (diagnostic plots)")


if __name__ == "__main__":
    main()