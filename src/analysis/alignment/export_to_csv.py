#!/usr/bin/env python3
"""
Export aligned HDF5 files to CSV format for maximum compatibility.
This creates a directory for each experiment with separate CSV files for each sensor.
"""

import pandas as pd
from pathlib import Path
import sys


def export_hdf5_to_csv(h5_file):
    """
    Export HDF5 file to CSV files.
    
    Args:
        h5_file: Path to HDF5 file
    """
    # Create output directory
    output_dir = h5_file.parent / h5_file.stem.replace('_aligned', '_csv')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nExporting {h5_file.name} to {output_dir}...")
    
    # Read and export each dataset
    with pd.HDFStore(h5_file, mode='r') as store:
        for key in store.keys():
            # Clean the key
            clean_key = key.lstrip('/')
            
            # Read the data
            df = store[key]
            
            # Export to CSV
            csv_file = output_dir / f"{clean_key}.csv"
            df.to_csv(csv_file, index=False)
            print(f"  Exported {clean_key}: {len(df)} rows to {csv_file.name}")
    
    # Create a summary file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Aligned data export from: {h5_file.name}\n")
        f.write(f"Export timestamp: {pd.Timestamp.now()}\n\n")
        f.write("Files in this directory:\n")
        for csv in sorted(output_dir.glob("*.csv")):
            f.write(f"  - {csv.name}\n")
    
    print(f"Export complete: {output_dir}")
    return output_dir


def main():
    """Export all aligned HDF5 files to CSV format."""
    aligned_dir = Path('aligned_data')
    
    if not aligned_dir.exists():
        print("Error: aligned_data directory not found!")
        sys.exit(1)
    
    # Find all HDF5 files
    h5_files = list(aligned_dir.glob('*.h5'))
    
    if not h5_files:
        print("No HDF5 files found!")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} HDF5 files to export")
    
    # Export each file
    exported_dirs = []
    for h5_file in h5_files:
        try:
            output_dir = export_hdf5_to_csv(h5_file)
            exported_dirs.append(output_dir)
        except Exception as e:
            print(f"Error exporting {h5_file}: {e}")
            continue
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"Exported {len(exported_dirs)} experiments to CSV format:")
    for d in exported_dirs:
        print(f"  - {d}")


if __name__ == "__main__":
    main()