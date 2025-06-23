#!/usr/bin/env python3
"""
Export specific static experiment HDF5 files to CSV format.
This script converts the aligned HDF5 files for experiments:
- 002_Setup
- 003_Waiting_for_departure  
- 010_Waiting_for_static_turns
"""

import h5py
import numpy as np
from pathlib import Path
import csv
import sys


def export_hdf5_to_csv_manual(h5_file_path, output_dir):
    """
    Export HDF5 file to CSV files without pandas dependency.
    
    Args:
        h5_file_path: Path to HDF5 file
        output_dir: Output directory for CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting {h5_file_path.name} to {output_dir}...")
    
    with h5py.File(h5_file_path, 'r') as f:
        # Iterate through all datasets
        for key in f.keys():
            dataset = f[key]
            
            # Check if it's a dataset
            if isinstance(dataset, h5py.Dataset):
                # Convert to numpy array
                data = dataset[:]
                
                # Get column names from attributes if available
                if 'columns' in dataset.attrs:
                    columns = dataset.attrs['columns']
                    if isinstance(columns, bytes):
                        columns = columns.decode('utf-8').split(',')
                    elif isinstance(columns, np.ndarray):
                        columns = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in columns]
                else:
                    # Generate default column names
                    if len(data.shape) == 2:
                        columns = [f'col_{i}' for i in range(data.shape[1])]
                    else:
                        columns = ['value']
                
                # Export to CSV
                csv_file = output_dir / f"{key}.csv"
                
                with open(csv_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    writer.writerow(columns)
                    
                    # Write data
                    if len(data.shape) == 1:
                        for value in data:
                            writer.writerow([value])
                    else:
                        for row in data:
                            writer.writerow(row)
                
                print(f"  Exported {key}: {len(data)} rows to {csv_file.name}")
    
    # Create summary file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Aligned data export from: {h5_file_path.name}\n")
        f.write(f"Export timestamp: {np.datetime64('now')}\n\n")
        f.write("Files in this directory:\n")
        for csv_file in sorted(output_dir.glob("*.csv")):
            f.write(f"  - {csv_file.name}\n")
    
    print(f"Export complete: {output_dir}")


def main():
    """Export static experiment HDF5 files to CSV format."""
    
    # Base directory for aligned data
    base_dir = Path("hovercraft_data_analysis/alignment_analysis/aligned_data")
    
    # Static experiments to export
    static_experiments = [
        "002_Setup",
        "003_Waiting_for_departure",
        "010_Waiting_for_static_turns"
    ]
    
    # Look for files in both morning and afternoon static directories
    h5_files = []
    for experiment in static_experiments:
        # Check static/morning
        pattern = f"static/morning/{experiment}_aligned.h5"
        file_path = base_dir / pattern
        if file_path.exists():
            h5_files.append(file_path)
        
        # Check static/afternoon
        pattern = f"static/afternoon/{experiment}_aligned.h5"
        file_path = base_dir / pattern
        if file_path.exists():
            h5_files.append(file_path)
    
    if not h5_files:
        print("No static experiment HDF5 files found!")
        print("Searched for:")
        for exp in static_experiments:
            print(f"  - static/morning/{exp}_aligned.h5")
            print(f"  - static/afternoon/{exp}_aligned.h5")
        return 1
    
    print(f"Found {len(h5_files)} static experiment HDF5 files to export:")
    for f in h5_files:
        print(f"  - {f.relative_to(base_dir)}")
    
    # Export each file
    exported_count = 0
    for h5_file in h5_files:
        try:
            # Create output directory next to HDF5 file
            output_dir = h5_file.parent / h5_file.stem.replace('_aligned', '_csv')
            export_hdf5_to_csv_manual(h5_file, output_dir)
            exported_count += 1
        except Exception as e:
            print(f"Error exporting {h5_file}: {e}")
            continue
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"Successfully exported {exported_count}/{len(h5_files)} static experiments to CSV format")
    
    return 0 if exported_count == len(h5_files) else 1


if __name__ == "__main__":
    sys.exit(main())