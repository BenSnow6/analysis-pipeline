#!/usr/bin/env python3
"""
Check WP2 results and summarize findings.
"""

import h5py
import numpy as np
from pathlib import Path
import pandas as pd

def check_hdf5_file(hdf5_path: Path):
    """Check contents of HDF5 file and print summary."""
    print(f"\n{'-'*60}")
    print(f"File: {hdf5_path.name}")
    print(f"{'-'*60}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check groups
        rpm_group = f['rpm_estimation']
        
        # Get data
        times = rpm_group['time'][:]
        rpms = rpm_group['rpm'][:]
        snr_db = rpm_group['snr_db'][:]
        valid = rpm_group['valid'][:]
        
        # Get attributes
        attrs = dict(rpm_group.attrs)
        
        print(f"Experiment: {attrs.get('experiment', 'N/A')}")
        print(f"Session: {attrs.get('session', 'N/A')}")
        print(f"Sensor: {attrs.get('sensor_id', 'N/A')}")
        print(f"Method: {attrs.get('method', 'N/A')}")
        
        print(f"\nData summary:")
        print(f"  Total frames: {len(times)}")
        print(f"  Valid frames: {np.sum(valid)} ({np.sum(valid)/len(times)*100:.1f}%)")
        print(f"  Duration: {times[-1] - times[0]:.1f} seconds")
        
        if np.sum(valid) > 0:
            valid_rpms = rpms[valid]
            valid_snrs = snr_db[valid]
            
            print(f"\nRPM statistics (valid frames only):")
            print(f"  Mean: {np.mean(valid_rpms):.0f} RPM")
            print(f"  Std: {np.std(valid_rpms):.0f} RPM")
            print(f"  Min: {np.min(valid_rpms):.0f} RPM")
            print(f"  Max: {np.max(valid_rpms):.0f} RPM")
            
            print(f"\nSNR statistics:")
            print(f"  Mean: {np.mean(valid_snrs):.1f} dB")
            print(f"  Min: {np.min(valid_snrs):.1f} dB")
            print(f"  Max: {np.max(valid_snrs):.1f} dB")
        
        # Check harmonics
        if 'harmonics' in rpm_group:
            harmonics = rpm_group['harmonics']
            print(f"\nHarmonics data available: {list(harmonics.keys())}")


def main():
    """Check all WP2 results."""
    print("WP-2 Results Summary")
    print("=" * 60)
    
    results_dir = Path(__file__).parent / 'results' / 'wp2' / 'afternoon'
    
    # Test experiments to check
    test_files = [
        "007_Fast_stbd_turn_1_Sensor_3_rpm.h5/007_Fast_stbd_turn_1_Sensor_3_rpm.h5",
        "003_Waiting_for_departure_Sensor_3_rpm.h5/003_Waiting_for_departure_Sensor_3_rpm.h5", 
        "026_Engine_rpm_sweep_Sensor_3_rpm.h5/026_Engine_rpm_sweep_Sensor_3_rpm.h5"
    ]
    
    for filepath in test_files:
        # Build full path
        full_path = results_dir / filepath
        if full_path.exists():
            check_hdf5_file(full_path)
        else:
            print(f"\nFile not found: {filepath}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()