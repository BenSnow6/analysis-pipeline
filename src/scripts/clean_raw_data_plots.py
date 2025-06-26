#!/usr/bin/env python3
"""
Clean up plot files from the raw data directory.
The /data/raw/ directory should contain only raw data files, not visualizations.
"""

import os
from pathlib import Path
import shutil

def count_and_list_plots(data_raw_dir: Path, dry_run: bool = True):
    """Find all PNG files in the raw data directory."""
    png_files = list(data_raw_dir.rglob("*.png"))
    
    print(f"Found {len(png_files)} PNG files in {data_raw_dir}")
    
    if dry_run:
        print("\nDRY RUN - No files will be deleted")
        print("\nSample of files that would be deleted:")
        for i, file in enumerate(png_files[:10]):
            print(f"  {file.relative_to(data_raw_dir)}")
        if len(png_files) > 10:
            print(f"  ... and {len(png_files) - 10} more files")
    
    # Analyze file types
    file_types = {}
    for file in png_files:
        if "GPS_Path" in file.name:
            ftype = "GPS Path plots"
        elif "GPS_Zoomed" in file.name:
            ftype = "GPS Zoomed plots"
        elif "GPS_Stitched" in file.name:
            ftype = "GPS Stitched plots"
        elif "GPS_" in file.name and any(x in file.name for x in ["Lat", "Lng", "Alt", "HDOP", "Bearing", "Speed"]):
            ftype = "GPS Parameter plots"
        elif "Cross_Plots" in file.name:
            ftype = "Cross-correlation plots"
        elif any(x in file.name for x in ["_accel.png", "_angle.png", "_gyro.png", "_mag.png"]):
            ftype = "IMU Sensor plots"
        else:
            ftype = "Other plots"
        
        file_types[ftype] = file_types.get(ftype, 0) + 1
    
    print("\nPlot types found:")
    for ftype, count in sorted(file_types.items()):
        print(f"  {ftype}: {count}")
    
    return png_files

def delete_plots(png_files: list, backup_dir: Path = None):
    """Delete or move PNG files."""
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nMoving {len(png_files)} files to {backup_dir}")
        
        for i, file in enumerate(png_files):
            if i % 100 == 0:
                print(f"  Processing file {i}/{len(png_files)}...")
            
            # Preserve directory structure in backup
            rel_path = file.relative_to(file.parents[3])  # relative to data/raw
            backup_path = backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(file), str(backup_path))
    else:
        print(f"\nDeleting {len(png_files)} PNG files...")
        
        for i, file in enumerate(png_files):
            if i % 100 == 0:
                print(f"  Deleting file {i}/{len(png_files)}...")
            file.unlink()
    
    print("\nDone!")

def main():
    """Main function."""
    repo_root = Path(__file__).parent
    data_raw_dir = repo_root / "data" / "raw"
    
    if not data_raw_dir.exists():
        print(f"Error: {data_raw_dir} does not exist")
        return
    
    print("Plot Cleanup Tool for /data/raw/")
    print("=" * 50)
    
    # First, do a dry run
    png_files = count_and_list_plots(data_raw_dir, dry_run=True)
    
    if not png_files:
        print("\nNo PNG files found. Directory is already clean!")
        return
    
    # Ask user what to do
    print("\n" + "=" * 50)
    print("Options:")
    print("1. Delete all PNG files (recommended)")
    print("2. Move PNG files to backup directory")
    print("3. Cancel")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        confirm = input(f"\nAre you sure you want to DELETE {len(png_files)} PNG files? (yes/no): ").strip().lower()
        if confirm == "yes":
            delete_plots(png_files)
        else:
            print("Cancelled.")
    elif choice == "2":
        backup_dir = repo_root / "data" / "plot_backup"
        confirm = input(f"\nMove {len(png_files)} PNG files to {backup_dir}? (yes/no): ").strip().lower()
        if confirm == "yes":
            delete_plots(png_files, backup_dir=backup_dir)
        else:
            print("Cancelled.")
    else:
        print("Cancelled.")

if __name__ == "__main__":
    main()