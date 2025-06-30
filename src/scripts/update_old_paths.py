#!/usr/bin/env python3
"""Update references to old data structure paths."""

from pathlib import Path
import re

def update_file(file_path: Path, replacements: dict[str, str]) -> bool:
    """Update a file with the given replacements."""
    try:
        content = file_path.read_text()
        original_content = content
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        if content != original_content:
            file_path.write_text(content)
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all references to old paths."""
    
    # Define replacements
    replacements = {
        # Old data structure paths
        '"02_Evaluation_Experiments"': '"data/raw"',
        "'02_Evaluation_Experiments'": "'data/raw'",
        "02_Evaluation_Experiments": "data/raw",
        
        # Timestamp results (now in processed data)
        "timestamp_analysis_results": "data/processed/timestamp",
    }
    
    # Files to update based on grep results
    files_to_update = [
        "src/analysis/orientation/add_gyro_to_csv.py",
        "src/analysis/orientation/analyze_gravity.py",
        "src/analysis/orientation/orientation_check.py",
        "src/analysis/timestamp/data_loader.py",
        "src/analysis/timestamp/main.py",
    ]
    
    updated_files = []
    
    for file_path_str in files_to_update:
        file_path = Path(file_path_str)
        if file_path.exists():
            if update_file(file_path, replacements):
                updated_files.append(file_path_str)
                print(f"Updated: {file_path_str}")
        else:
            print(f"File not found: {file_path_str}")
    
    # Also update markdown files
    md_files = [
        "src/analysis/orientation/orientation_analysis_status.md",
    ]
    
    for file_path_str in md_files:
        file_path = Path(file_path_str)
        if file_path.exists():
            if update_file(file_path, replacements):
                updated_files.append(file_path_str)
                print(f"Updated: {file_path_str}")
    
    print(f"\nTotal files updated: {len(updated_files)}")
    
    if updated_files:
        print("\nPlease review the changes and test that the code still works correctly.")
        print("You may need to update the logic to handle the new directory structure.")

if __name__ == "__main__":
    main()