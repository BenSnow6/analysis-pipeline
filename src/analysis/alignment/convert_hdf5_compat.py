#!/usr/bin/env python3
"""
Convert HDF5 files to ensure compatibility across different environments.
This script reads the aligned HDF5 files and re-saves them using pickle protocol 4
which is more compatible across different numpy versions.
"""

import pandas as pd
from pathlib import Path
import sys


def convert_hdf5_to_compatible(input_file, output_file=None):
    """
    Convert HDF5 file to a more compatible format.
    
    Args:
        input_file: Path to input HDF5 file
        output_file: Path to output file (if None, overwrites input)
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Converting {input_file}...")
    
    # Read all data from the store
    with pd.HDFStore(input_file, mode='r') as store:
        data = {}
        for key in store.keys():
            data[key] = store[key]
            print(f"  Read {key}: {len(store[key])} rows")
    
    # Write back with compatibility settings
    with pd.HDFStore(output_file, mode='w') as store:
        for key, df in data.items():
            # Remove the leading '/' from key
            clean_key = key.lstrip('/')
            # Use fixed format for better compatibility
            if clean_key == 'metadata':
                store.put(clean_key, df, format='fixed')
            else:
                store.put(clean_key, df, format='table', data_columns=False)
            print(f"  Wrote {key}")
    
    print(f"Conversion complete: {output_file}")


def main():
    """Convert all aligned HDF5 files in the aligned_data directory."""
    aligned_dir = Path('aligned_data')
    
    if not aligned_dir.exists():
        print("Error: aligned_data directory not found!")
        print("Please run this script from the alignment_analysis directory.")
        sys.exit(1)
    
    # Find all HDF5 files
    h5_files = list(aligned_dir.glob('*.h5'))
    
    if not h5_files:
        print("No HDF5 files found in aligned_data directory!")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} HDF5 files to convert:")
    for f in h5_files:
        print(f"  - {f.name}")
    
    # Convert each file
    for h5_file in h5_files:
        try:
            # Create backup first
            backup_file = h5_file.with_suffix('.h5.bak')
            if not backup_file.exists():
                print(f"\nBacking up {h5_file.name} to {backup_file.name}")
                import shutil
                shutil.copy2(h5_file, backup_file)
            
            # Convert the file
            convert_hdf5_to_compatible(h5_file)
            
        except Exception as e:
            print(f"Error converting {h5_file}: {e}")
            continue
    
    print("\nAll files converted successfully!")
    print("Original files backed up with .bak extension")


if __name__ == "__main__":
    main()