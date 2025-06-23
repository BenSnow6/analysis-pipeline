#!/usr/bin/env python3
"""
Run orientation analysis on static experiments only.
Uses the truly static experiments from all_expts folder for best results.
"""

import yaml
import subprocess
import sys
from pathlib import Path
import argparse


def load_manifest(manifest_path):
    """Load experiment manifest from YAML file."""
    with open(manifest_path, 'r') as f:
        return yaml.safe_load(f)


def run_static_alignment(static_experiments, session, dry_run=False):
    """
    Run alignment on static experiments if not already done.
    
    Args:
        static_experiments: List of static experiment dicts
        session: 'morning' or 'afternoon'
        dry_run: If True, just print commands
    """
    print(f"\n{'='*60}")
    print(f"Aligning {session.upper()} static experiments")
    print(f"{'='*60}")
    
    base_path = Path(__file__).parent
    
    # First check if these experiments are already aligned
    aligned_data_dir = base_path / "alignment_analysis" / "aligned_data" / "static" / session
    
    experiments_to_align = []
    for exp in static_experiments:
        exp_name = exp['name']
        if not (aligned_data_dir / f"{exp_name}_aligned.h5").exists():
            experiments_to_align.append(exp_name)
    
    if not experiments_to_align:
        print("All static experiments already aligned!")
        return True
    
    print(f"Need to align: {experiments_to_align}")
    
    # Create output directory
    aligned_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command - use full path to find experiments
    cmd = [
        'python', 
        'alignment_analysis/run_alignment.py',
        '-e'
    ] + experiments_to_align + [
        '-o', str(aligned_data_dir),
        '-b', str(base_path.parent)  # Use parent to search in all_expts
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
    
    # Run additional data alignment
    print("\nRunning additional data alignment...")
    # Create a temporary script that processes static experiments
    script_content = f'''
import sys
sys.path.append("{base_path / 'alignment_analysis'}")
from align_additional_data import process_experiment
from pathlib import Path

base_path = Path("{base_path.parent}")
aligned_path = Path("{aligned_data_dir}")
experiments = {experiments_to_align}

for exp in experiments:
    process_experiment(exp, base_path / "02_Evaluation_Experiments", aligned_path)
'''
    
    if not dry_run:
        temp_script = base_path / "temp_align_static.py"
        temp_script.write_text(script_content)
        
        result = subprocess.run(['python', str(temp_script)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        temp_script.unlink()  # Clean up
    
    return True


def run_orientation_validation(static_experiments, session, dry_run=False):
    """
    Run orientation validation on static experiments.
    
    Args:
        static_experiments: List of static experiment dicts  
        session: 'morning' or 'afternoon'
        dry_run: If True, just print commands
    """
    print(f"\n{'='*60}")
    print(f"Running orientation validation for {session.upper()}")
    print(f"{'='*60}")
    
    base_path = Path(__file__).parent
    exp_names = [exp['name'] for exp in static_experiments]
    
    # Build command
    cmd = [
        'python',
        'orientation_analysis/run_orientation.py',
        '-e'
    ] + exp_names + [
        '-d', str(base_path / "alignment_analysis" / "aligned_data" / "static" / session),
        '-o', str(base_path / "orientation_analysis" / "validation_results" / "static" / session)
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    if not dry_run:
        try:
            result = subprocess.run(cmd, cwd=base_path, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    else:
        print("[DRY RUN] Command not executed")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run orientation analysis on static experiments"
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
        help="Skip alignment step (if already done)"
    )
    
    args = parser.parse_args()
    
    # Load manifest
    base_path = Path(__file__).parent
    manifest = load_manifest(base_path / args.manifest)
    
    print(f"Using static experiments from: {args.manifest}")
    print(f"Sessions to process: {args.sessions}")
    
    # Process each session
    all_success = True
    for session in args.sessions:
        static_exps = manifest['static_experiments'][session]
        
        print(f"\n{session.upper()} SESSION:")
        print(f"  Static experiments: {[e['name'] for e in static_exps]}")
        
        # Step 1: Align static experiments if needed
        if not args.skip_alignment:
            success = run_static_alignment(static_exps, session, args.dry_run)
            if not success and not args.dry_run:
                print(f"ERROR: Alignment failed for {session}")
                all_success = False
                continue
        
        # Step 2: Run orientation validation
        success = run_orientation_validation(static_exps, session, args.dry_run)
        if not success and not args.dry_run:
            print(f"ERROR: Orientation validation failed for {session}")
            all_success = False
    
    print("\n" + "="*60)
    if all_success:
        print("Static orientation analysis complete!")
        print("Results saved in: orientation_analysis/validation_results/static/")
    else:
        print("Some analyses failed. Check output above.")
    print("="*60)
    
    # Summary of what to do with results
    print("\nNEXT STEPS:")
    print("1. Review orientation validation results for each session")
    print("2. Use validated rotation matrices for all experiments in that session")
    print("3. Apply morning bias corrections to morning experiments only")
    print("4. Apply afternoon bias corrections to afternoon experiments only")
    print("5. Never mix data between morning and afternoon sessions!")


if __name__ == "__main__":
    main()