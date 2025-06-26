#!/usr/bin/env python3
"""
Align additional sensor data (gyro, angle, mag) for all experiments.
Processes morning and afternoon sessions separately.
"""

from pathlib import Path
from src.analysis.alignment.align_additional_data import process_experiment
from tqdm import tqdm


def main():
    """Process all aligned experiments."""
    
    from src.config.paths import DATA_DIR
    base_data_path = DATA_DIR
    aligned_data_path = Path(__file__).parent / "aligned_data"
    
    # Find all CSV directories
    csv_dirs = list(aligned_data_path.rglob("*_csv"))
    
    # Filter to only experiment directories (not subdirectories)
    experiment_dirs = [d for d in csv_dirs if d.parent.name in ['aligned_data', 'morning', 'afternoon']]
    
    print(f"Found {len(experiment_dirs)} experiments to process for additional data")
    
    # Group by session
    morning_exps = [d for d in experiment_dirs if 'morning' in str(d)]
    afternoon_exps = [d for d in experiment_dirs if 'afternoon' in str(d)]
    root_exps = [d for d in experiment_dirs if d.parent.name == 'aligned_data']
    
    print(f"\nMorning experiments: {len(morning_exps)}")
    print(f"Afternoon experiments: {len(afternoon_exps)}")
    print(f"Root experiments: {len(root_exps)}")
    
    success_count = 0
    total_count = 0
    
    # Process each group
    for session_name, session_dirs in [("Morning", morning_exps), 
                                       ("Afternoon", afternoon_exps), 
                                       ("Root", root_exps)]:
        if session_dirs:
            print(f"\n{'='*60}")
            print(f"Processing {session_name} experiments")
            print(f"{'='*60}")
            
            for exp_dir in tqdm(session_dirs, desc=f"{session_name} experiments"):
                exp_name = exp_dir.name.replace("_csv", "")
                
                # Determine aligned data path (parent of CSV dir)
                aligned_path = exp_dir.parent
                
                if process_experiment(exp_name, base_data_path, aligned_path):
                    success_count += 1
                total_count += 1
    
    print(f"\n{'='*60}")
    print(f"Additional alignment complete: {success_count}/{total_count} experiments processed")
    print(f"{'='*60}")
    
    if success_count < total_count:
        print(f"\nWARNING: {total_count - success_count} experiments failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())