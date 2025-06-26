#!/usr/bin/env python3
"""
Standalone script to run timestamp analysis without all dependencies.
This provides a preview of what the analysis would show.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# No need to modify sys.path with proper imports

def check_dependencies():
    """Check if required dependencies are available."""
    required = ['pandas', 'numpy', 'matplotlib', 'yaml', 'seaborn']
    missing = []
    
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print("ERROR: Missing required Python packages:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True

def analyze_experiments_preview():
    """Preview of experiment analysis without full dependencies."""
    print("="*80)
    print("TIMESTAMP ANALYSIS PREVIEW")
    print("="*80)
    
    # Define the data path
    from src.config.paths import DATA_DIR
    data_path = DATA_DIR
    
    if not data_path.exists():
        print(f"ERROR: Data path not found: {data_path}")
        return
    
    # Find all experiments
    experiments = []
    for root, dirs, files in os.walk(data_path):
        root_path = Path(root)
        
        # Check if this directory contains GPS and IMU subdirectories
        has_gps = (root_path / "GPS").is_dir()
        has_imu = (root_path / "IMU").is_dir()
        
        if has_gps and has_imu:
            relative_path = root_path.relative_to(data_path)
            experiments.append({
                'name': str(relative_path).replace(os.sep, '/'),
                'path': str(root_path),
                'gps_files': len(list((root_path / "GPS").glob("*.csv"))),
                'imu_sensors': len(list((root_path / "IMU").iterdir()))
            })
    
    print(f"\nFound {len(experiments)} experiments to analyze:\n")
    
    # Create summary report
    summary = {
        'analysis_time': datetime.now().isoformat(),
        'total_experiments': len(experiments),
        'experiments': []
    }
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
        print(f"   Path: {exp['path']}")
        print(f"   GPS files: {exp['gps_files']}")
        print(f"   IMU sensors: {exp['imu_sensors']}")
        
        # Add to summary
        summary['experiments'].append({
            'name': exp['name'],
            'gps_files': exp['gps_files'],
            'imu_sensors': exp['imu_sensors']
        })
        
        # Show first few files as example
        if i == 1:
            print("\n   Example files:")
            gps_dir = Path(exp['path']) / "GPS"
            for f in list(gps_dir.glob("*.csv"))[:1]:
                print(f"     GPS: {f.name}")
            
            imu_dir = Path(exp['path']) / "IMU"
            for sensor_dir in list(imu_dir.iterdir())[:2]:
                if sensor_dir.is_dir():
                    files = list(sensor_dir.glob("*.csv"))[:1]
                    if files:
                        print(f"     IMU/{sensor_dir.name}: {files[0].name}")
        
        print()
    
    # Save summary
    output_dir = Path("timestamp_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Install required dependencies:")
    print("   pip install pandas numpy matplotlib seaborn pyyaml")
    print("\n2. Run full analysis:")
    print("   python3 -m hovercraft_analysis.analysis.timestamp --all")
    print("\n3. View results in: timestamp_analysis_results/")
    print("="*80)

def main():
    """Main entry point."""
    print("Checking for required dependencies...")
    
    if check_dependencies():
        print("✓ All dependencies available. Running full analysis...")
        # Import and run the actual analysis
        from src.analysis.timestamp.main import main as run_analysis
        
        # Simulate command line args for --all
        sys.argv = [sys.argv[0], '--all', '--output', 'timestamp_analysis_results']
        run_analysis()
    else:
        print("\nRunning preview mode (no dependencies required)...")
        analyze_experiments_preview()

if __name__ == "__main__":
    main()