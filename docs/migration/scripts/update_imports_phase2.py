#!/usr/bin/env python3
"""
Update all imports from 'code.' to 'hovercraft_analysis.' for Phase 2 migration.
"""
import os
import re
from pathlib import Path


def update_imports_in_file(filepath: Path) -> bool:
    """Update imports in a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
    
    original_content = content
    
    # Pattern replacements
    replacements = [
        # Direct code. imports
        (r'from code\.', 'from src.'),
        (r'import code\.', 'import src.'),
        
        # Specific module mappings
        (r'from code\.alignment_analysis', 'from src.analysis.alignment'),
        (r'from code\.orientation_analysis', 'from src.analysis.orientation'),
        (r'from code\.timestamp_analysis', 'from src.analysis.timestamp'),
        (r'from code\.rpm_estimation', 'from src.analysis.rpm'),
        (r'from code\.dashboard_app', 'from src.apps.dashboard'),
        (r'from code\.scripts', 'from src.scripts'),
        (r'from code\.config\.paths', 'from src.core.paths'),
        (r'from code\.config', 'from src.core'),
        
        # Update any remaining code. references
        (r'code\.config\.paths', 'hovercraft_analysis.core.paths'),
        (r'code\.alignment_analysis', 'hovercraft_analysis.analysis.alignment'),
        (r'code\.orientation_analysis', 'hovercraft_analysis.analysis.orientation'),
        (r'code\.timestamp_analysis', 'hovercraft_analysis.analysis.timestamp'),
        (r'code\.rpm_estimation', 'hovercraft_analysis.analysis.rpm'),
        (r'code\.dashboard_app', 'hovercraft_analysis.apps.dashboard'),
        
        # Update relative imports within moved modules
        (r'from \.\.config import', 'from src.core import'),
        (r'from \.\.config\.paths import', 'from src.core.paths import'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Special handling for files that might use old paths
    if 'alignment_analysis/aligned_data' in content:
        content = content.replace(
            'alignment_analysis/aligned_data',
            'processed/aligned'
        )
    
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {filepath}")
            return True
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
            return False
    
    return False


def update_all_imports():
    """Update imports in all Python files."""
    # Directories to process
    directories = [
        Path("src/hovercraft_analysis"),
        Path("tests"),
    ]
    
    updated_count = 0
    error_count = 0
    
    for directory in directories:
        if not directory.exists():
            print(f"Directory not found: {directory}")
            continue
        
        for py_file in directory.rglob("*.py"):
            # Skip __pycache__ directories
            if "__pycache__" in str(py_file):
                continue
            
            try:
                if update_imports_in_file(py_file):
                    updated_count += 1
            except Exception as e:
                print(f"Error processing {py_file}: {e}")
                error_count += 1
    
    print(f"\nSummary:")
    print(f"  Files updated: {updated_count}")
    print(f"  Errors: {error_count}")


def main():
    """Main entry point."""
    print("Updating imports for Phase 2 migration...")
    print("Converting 'code.' imports to 'hovercraft_analysis.'")
    print("-" * 60)
    
    update_all_imports()
    
    print("\nNext steps:")
    print("1. Review the changes")
    print("2. Test imports with: python -m pytest tests/")
    print("3. Install package with: pip install -e .")


if __name__ == "__main__":
    main()