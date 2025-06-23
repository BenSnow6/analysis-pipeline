#!/usr/bin/env python3
"""
Process all experiments from the experiment manifest.
Handles morning/afternoon sessions separately and runs alignment on all evaluation experiments.
"""

import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse


def load_manifest(manifest_path):
    """Load experiment manifest from YAML file."""
    with open(manifest_path, 'r') as f:
        return yaml.safe_load(f)


def run_alignment_batch(experiments, session, base_path, output_dir, dry_run=False):
    """
    Run alignment on a batch of experiments.
    
    Args:
        experiments: List of experiment dicts
        session: 'morning' or 'afternoon'
        base_path: Base path for experiments
        output_dir: Output directory
        dry_run: If True, just print commands
    """
    print(f"\n{'='*60}")
    print(f"Processing {session.upper()} experiments")
    print(f"{'='*60}")
    
    # Build experiment names list
    exp_names = [exp['name'] for exp in experiments]
    
    # Build command - need to specify base path for experiments
    cmd = [
        'python', 
        'alignment_analysis/run_alignment.py',
        '-e'
    ] + exp_names + [
        '-o', str(output_dir / session),
        '-b', str(base_path.parent)  # Point to parent directory containing 02_Evaluation_Experiments
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    if not dry_run:
        try:
            result = subprocess.run(cmd, cwd=base_path, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.returncode != 0:
                print(f"ERROR: Alignment failed with return code {result.returncode}")
                return False
        except Exception as e:
            print(f"ERROR running alignment: {e}")
            return False
    else:
        print("[DRY RUN] Command not executed")
    
    return True


def run_additional_alignment(session, aligned_data_dir, dry_run=False):
    """Run align_additional_data.py for a specific session."""
    print(f"\nRunning additional data alignment for {session}...")
    
    # Update align_additional_data.py to process the right experiments
    script_path = Path("alignment_analysis/align_additional_data_session.py")
    
    # Create session-specific version of the script
    template = '''#!/usr/bin/env python3
"""
Align additional sensor data for {session} session.
Auto-generated from process_all_experiments.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from align_additional_data import align_sensor_data, process_experiment

def main():
    base_data_path = Path(__file__).parent.parent.parent / "02_Evaluation_Experiments"
    aligned_data_path = Path(__file__).parent / "aligned_data" / "{session}"
    
    # List all experiments in aligned_data_path
    experiments = []
    for exp_dir in aligned_data_path.glob("*_csv"):
        exp_name = exp_dir.name.replace("_csv", "")
        experiments.append(exp_name)
    
    print(f"Found {{len(experiments)}} {session} experiments to process")
    
    success_count = 0
    for exp in experiments:
        print(f"\\nProcessing {{exp}}...")
        if process_experiment(exp, base_data_path, aligned_data_path):
            success_count += 1
    
    print(f"\\n{'='*60}")
    print(f"Additional alignment complete: {{success_count}}/{{len(experiments)}} experiments processed")
    print(f"{'='*60}")
    
    return 0 if success_count == len(experiments) else 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    if not dry_run:
        script_path.write_text(template.format(session=session))
        
        # Run the script
        cmd = ['python', str(script_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    else:
        print(f"[DRY RUN] Would create and run {script_path}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Process all experiments from manifest"
    )
    parser.add_argument(
        "--manifest",
        default="experiment_manifest.yaml",
        help="Path to experiment manifest file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=["morning", "afternoon"],
        default=["morning", "afternoon"],
        help="Which sessions to process"
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip main alignment (if already done)"
    )
    parser.add_argument(
        "--skip-additional",
        action="store_true",
        help="Skip additional data alignment"
    )
    
    args = parser.parse_args()
    
    # Load manifest
    base_path = Path(__file__).parent
    manifest = load_manifest(base_path / args.manifest)
    
    # Create output directory
    output_dir = base_path / "alignment_analysis" / "aligned_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing experiments from: {args.manifest}")
    print(f"Output directory: {output_dir}")
    print(f"Sessions to process: {args.sessions}")
    
    # Process each session
    for session in args.sessions:
        experiments = manifest['evaluation_experiments'][session]
        
        print(f"\n{session.upper()} SESSION:")
        print(f"  Found {len(experiments)} experiments")
        
        # Step 1: Run main alignment
        if not args.skip_alignment:
            success = run_alignment_batch(
                experiments, 
                session, 
                base_path,
                output_dir,
                args.dry_run
            )
            if not success and not args.dry_run:
                print(f"ERROR: Alignment failed for {session}")
                continue
        else:
            print(f"  Skipping main alignment for {session}")
        
        # Step 2: Run additional data alignment
        if not args.skip_additional:
            success = run_additional_alignment(
                session,
                output_dir,
                args.dry_run
            )
            if not success and not args.dry_run:
                print(f"ERROR: Additional alignment failed for {session}")
    
    print("\n" + "="*60)
    print("All processing complete!")
    print(f"Aligned data saved in: {output_dir}/morning and {output_dir}/afternoon")
    print("="*60)


if __name__ == "__main__":
    main()