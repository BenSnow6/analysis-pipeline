#!/usr/bin/env python3
"""
Export all aligned HDF5 files to CSV format, including morning/afternoon subdirectories.
"""

import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm


def export_hdf5_to_csv(hdf5_path: Path, output_dir: Path):
    """Export a single HDF5 file to CSV format."""
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open HDF5 file
        with pd.HDFStore(str(hdf5_path), mode='r') as store:
            # Export each key/sensor
            for key in store.keys():
                # Clean key name (remove leading '/')
                sensor_name = key.strip('/')
                
                # Read data
                df = store[key]
                
                # Save to CSV
                csv_path = output_dir / f"{sensor_name}.csv"
                df.to_csv(csv_path, index=False)
                print(f"  Exported {sensor_name}: {len(df)} rows to {sensor_name}.csv")
        
        # Create summary file
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Aligned data exported from: {hdf5_path.name}\n")
            f.write(f"Export timestamp: {pd.Timestamp.now()}\n")
        
        return True
        
    except Exception as e:
        print(f"  ERROR exporting {hdf5_path.name}: {e}")
        return False


def main():
    """Export all HDF5 files to CSV format."""
    base_dir = Path(__file__).parent / "aligned_data"
    
    # Find all HDF5 files recursively
    hdf5_files = list(base_dir.rglob("*.h5"))
    
    if not hdf5_files:
        print("No HDF5 files found to export")
        return 1
    
    print(f"Found {len(hdf5_files)} HDF5 files to export\n")
    
    successful = 0
    for hdf5_path in tqdm(hdf5_files, desc="Exporting"):
        # Create output directory next to HDF5 file
        output_dir = hdf5_path.parent / hdf5_path.stem.replace('_aligned', '_csv')
        
        print(f"\nExporting {hdf5_path.relative_to(base_dir)} to {output_dir.relative_to(base_dir)}...")
        
        if export_hdf5_to_csv(hdf5_path, output_dir):
            successful += 1
            print(f"Export complete: {output_dir.relative_to(base_dir)}")
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"Successfully exported {successful}/{len(hdf5_files)} experiments to CSV format")
    
    return 0 if successful == len(hdf5_files) else 1


if __name__ == "__main__":
    sys.exit(main())