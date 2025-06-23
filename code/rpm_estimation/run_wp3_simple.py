#!/usr/bin/env python3
"""
Simple WP-3 runner to process experiments that have WP-2 results.
"""

import subprocess
import sys
from pathlib import Path

def run_wp3_for_experiment(experiment, session="afternoon"):
    """Run WP-3 processing for a single experiment."""
    print(f"\n{'='*60}")
    print(f"Processing {experiment} ({session} session)")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "-m", "code.rpm_estimation.wp3_process",
        "--session", session,
        "--experiment", experiment
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    
    if result.returncode == 0:
        print(f"✓ Successfully processed {experiment}")
    else:
        print(f"✗ Failed to process {experiment}")
    
    return result.returncode == 0

def main():
    # Experiments that have WP-2 results
    experiments_with_wp2 = [
        "003_Waiting_for_departure",
        "007_Fast_stbd_turn_1", 
        "026_Engine_rpm_sweep"
    ]
    
    # Check which ones have aligned data
    base_path = Path(__file__).parent.parent.parent / 'hovercraft_data_analysis' / 'alignment_analysis' / 'aligned_data'
    afternoon_dir = base_path / 'afternoon'
    
    available_experiments = []
    for exp in experiments_with_wp2:
        exp_dir = afternoon_dir / f"{exp}_csv"
        if exp_dir.exists():
            available_experiments.append(exp)
            print(f"✓ Found aligned data for: {exp}")
        else:
            print(f"✗ No aligned data for: {exp}")
    
    print(f"\nWill process {len(available_experiments)} experiments with both WP-2 and aligned data")
    
    # Process each available experiment
    successful = 0
    for exp in available_experiments:
        if run_wp3_for_experiment(exp):
            successful += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {successful}/{len(available_experiments)} experiments processed successfully")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()