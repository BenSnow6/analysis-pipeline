#!/usr/bin/env python3
"""
Command-line interface for running data alignment on hovercraft experiments.

Usage:
    python run_alignment.py -e 007_Fast_stbd_turn_1 -o aligned_data/
    python run_alignment.py -e 016_Straight_cruise_1 --dry-run
"""

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

from align import DataAligner


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Align multi-rate sensor data for hovercraft experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Align a single experiment
  python run_alignment.py -e 007_Fast_stbd_turn_1
  
  # Specify output directory
  python run_alignment.py -e 016_Straight_cruise_1 -o custom_output/
  
  # Dry run to check performance
  python run_alignment.py -e 021_Quarter_turn_port --dry-run
  
  # Process multiple experiments
  python run_alignment.py -e 007_Fast_stbd_turn_1 016_Straight_cruise_1 021_Quarter_turn_port
        """
    )
    
    parser.add_argument(
        '-e', '--experiments',
        nargs='+',
        required=True,
        help='Experiment name(s) to process'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('aligned_data'),
        help='Output directory for aligned HDF5 files (default: aligned_data/)'
    )
    
    parser.add_argument(
        '-b', '--base-path',
        type=Path,
        default=Path.cwd(),
        help='Base path to search for experiments (default: current directory)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run alignment without saving files, print statistics only'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=Path,
        help='Path to custom configuration file (default: use built-in config)'
    )
    
    return parser.parse_args()


def process_experiment(aligner, experiment_name, base_path, output_dir, dry_run=False):
    """
    Process a single experiment.
    
    Args:
        aligner: DataAligner instance
        experiment_name: Name of the experiment
        base_path: Base path to search for experiments
        output_dir: Output directory for aligned files
        dry_run: If True, don't save files
        
    Returns:
        Tuple of (success, elapsed_time, num_samples)
    """
    print(f"\n{'='*60}")
    print(f"Processing experiment: {experiment_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Load data with progress bar
        print("Loading sensor data...")
        sensor_data = aligner.load_experiment_data(experiment_name, base_path)
        
        if not sensor_data:
            print(f"Error: No sensor data found for {experiment_name}")
            return False, 0, 0
        
        # Align sensors
        print("\nAligning sensors...")
        aligned_data = aligner.align_all_sensors(sensor_data)
        
        # Calculate total samples
        total_samples = sum(len(df) for df in aligned_data.values())
        
        if not dry_run:
            # Save aligned data
            output_path = output_dir / f"{experiment_name}_aligned.h5"
            print(f"\nSaving aligned data to {output_path}")
            aligner.save_aligned_data(output_path)
        else:
            print("\n[DRY RUN] Skipping file save")
        
        elapsed = time.time() - start_time
        
        # Print summary
        print(f"\nAlignment complete!")
        print(f"  Total time: {elapsed:.3f} seconds")
        print(f"  Sensors aligned: {len(aligned_data)}")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Performance: {total_samples/elapsed:,.0f} samples/second")
        
        # Check performance target
        if elapsed > 1.0:
            print(f"  ⚠️  Warning: Exceeded 1-second target (took {elapsed:.3f}s)")
        else:
            print(f"  ✓ Within 1-second target")
        
        return True, elapsed, total_samples
        
    except Exception as e:
        print(f"Error processing {experiment_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0, 0


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create output directory if needed
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize aligner
    print("Initializing DataAligner...")
    aligner = DataAligner(args.config)
    
    # Process experiments with progress bar
    results = []
    with tqdm(total=len(args.experiments), desc="Experiments", unit="exp") as pbar:
        for experiment in args.experiments:
            success, elapsed, samples = process_experiment(
                aligner, experiment, args.base_path, args.output_dir, args.dry_run
            )
            results.append((experiment, success, elapsed, samples))
            pbar.update(1)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for _, success, _, _ in results if success)
    total_time = sum(elapsed for _, _, elapsed, _ in results)
    total_samples = sum(samples for _, _, _, samples in results)
    
    print(f"Experiments processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Total samples: {total_samples:,}")
    
    if not args.dry_run and successful > 0:
        print(f"\nAligned data saved to: {args.output_dir}/")
    
    # Exit with error code if any experiments failed
    if successful < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()