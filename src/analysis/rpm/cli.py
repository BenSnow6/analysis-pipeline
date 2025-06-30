"""
Command-line interface for RPM estimation.

This module provides the CLI entry point for running RPM estimation
on hovercraft IMU data.
"""

import argparse
import sys
import logging
from pathlib import Path
import json
from typing import List, Optional
import pandas as pd
from .io import load_config, list_available_experiments
from .preprocess import process_experiment_wp1, process_sensor_wp1
from .logging_config import setup_logging as setup_structured_logging


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='RPM Estimation from IMU Vibration Data',
        epilog='Example: python -m rpm_estimation.cli --wp 1 --exp 026_Engine_rpm_sweep --session afternoon'
    )
    
    # Work package selection
    parser.add_argument('--wp', type=int, choices=[0, 1, 2, 3, 4, 5, 6],
                       default=1, help='Work package to execute (default: 1)')
    
    # Experiment selection
    parser.add_argument('--exp', 
                       help='Experiment name (e.g., 026_Engine_rpm_sweep)')
    parser.add_argument('--session', choices=['morning', 'afternoon'],
                       help='Data collection session')
    
    # Batch processing
    parser.add_argument('--all', action='store_true',
                       help='Process all experiments for the session')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    
    # Method selection (for future WPs)
    parser.add_argument('--method', choices=['welch', 'stft', 'both'],
                       default='welch', help='Estimation method (default: welch)')
    
    # Configuration
    parser.add_argument('--config', default='rpm_config.yaml',
                       type=Path, help='Path to configuration file')
    
    # Output options
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for results (defaults to aligned_data)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    # Sensor selection
    parser.add_argument('--sensors', nargs='+',
                       help='Sensors to process (defaults to config)')
    
    # Processing options
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate processing results')
    
    # Validation
    parser.add_argument('--validate', action='store_true',
                       help='Run validation tests')
    parser.add_argument('--include-synthetic', action='store_true',
                       help='Include synthetic test in validation')
    
    # Logging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-format', choices=['json', 'text'],
                       default='text', help='Log output format')
    parser.add_argument('--log-file', type=Path,
                       help='Log file path')
    
    # Debugging
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without running')
    
    # WP-3 specific options
    wp3_group = parser.add_argument_group('WP-3 STFT options')
    wp3_group.add_argument('--snr-threshold', type=float,
                          help='Override SNR threshold for STFT (default: from config)')
    wp3_group.add_argument('--no-smoothing', action='store_true',
                          help='Disable RPM smoothing for high-rate regions')
    wp3_group.add_argument('--edge-padding', choices=['mirror', 'wrap', 'trim'],
                          help='Edge padding method for STFT (default: mirror)')
    
    # WP-4 specific options
    wp4_group = parser.add_argument_group('WP-4 Fusion options')
    wp4_group.add_argument('--fusion-strategy', 
                          choices=['snr_max', 'median', 'weighted'],
                          default='snr_max',
                          help='Fusion strategy (default: snr_max)')
    wp4_group.add_argument('--min-sensors', type=int,
                          help='Minimum sensors required for valid estimate')
    wp4_group.add_argument('--interpolation-window', type=float,
                          help='Maximum gap to interpolate in seconds (default: 5.0)')
    wp4_group.add_argument('--save-fusion-intermediate', action='store_true',
                          help='Save intermediate fusion data')
    
    return parser


def setup_logging(debug: bool = False):
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def validate_args(args):
    """Validate command-line arguments."""
    # Check if config file exists
    if not args.config.exists():
        # If default config doesn't exist in current dir, check standard locations
        module_dir = Path(__file__).parent
        project_root = module_dir.parent.parent.parent
        
        # Check multiple standard locations
        possible_locations = [
            module_dir / 'rpm_config.yaml',  # In the module directory
            project_root / 'config' / 'processing' / 'rpm_config.yaml',  # Standard config location
            project_root / 'config' / 'rpm_config.yaml',  # Alternative location
        ]
        
        for location in possible_locations:
            if location.exists():
                args.config = location
                break
        else:
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Create output directory if needed
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    return args


def run_wp1(args, config, logger):
    """Execute Work Package 1: Raw data audit and orientation."""
    
    if args.validate:
        logger.info("Running WP-1 validation suite")
        # TODO: Implement validation tests
        return run_wp1_validation(args, config, logger)
    
    if args.list:
        # List available experiments
        experiments = list_available_experiments()
        logger.info("Available experiments:")
        for session, exp_list in experiments.items():
            logger.info(f"\n{session.capitalize()}:")
            for exp in sorted(exp_list):
                logger.info(f"  - {exp}")
        return 0
    
    # Validate required arguments
    if not args.all and (not args.exp or not args.session):
        logger.error("Either --all or both --exp and --session must be specified")
        return 1
    
    # Process experiments
    if args.all:
        if not args.session:
            logger.error("--session must be specified with --all")
            return 1
            
        # Get all experiments for session
        experiments = list_available_experiments()
        exp_list = experiments.get(args.session, [])
        
        if not exp_list:
            logger.error(f"No experiments found for {args.session} session")
            return 1
        
        logger.info(f"Processing {len(exp_list)} experiments for {args.session} session")
        
        # Process each experiment
        all_results = {}
        for exp in exp_list:
            logger.info(f"\nProcessing {exp}...")
            results = process_experiment_wp1(
                exp, args.session, config,
                sensors=args.sensors,
                output_base=args.output_dir,
                parallel=not args.no_parallel
            )
            all_results[exp] = results
        
        # Summary
        total_sensors = sum(r['sensors_processed'] for r in all_results.values())
        total_successful = sum(r['successful'] for r in all_results.values())
        
        logger.info(f"\nBatch processing complete:")
        logger.info(f"  Experiments: {len(all_results)}")
        logger.info(f"  Total sensors: {total_sensors}")
        logger.info(f"  Successful: {total_successful}")
        
    else:
        # Process single experiment
        if args.dry_run:
            logger.info("DRY RUN - No processing will be performed")
            logger.info(f"Would process: {args.exp} ({args.session})")
            logger.info(f"Would process sensors: {args.sensors or 'from config'}")
            return 0
        
        results = process_experiment_wp1(
            args.exp, args.session, config,
            sensors=args.sensors,
            output_base=args.output_dir,
            parallel=not args.no_parallel
        )
        
        # Log results
        logger.info(f"\nProcessing complete for {args.exp}:")
        logger.info(f"  Sensors processed: {results['sensors_processed']}")
        logger.info(f"  Successful: {results['successful']}")
        logger.info(f"  Failed: {results['failed']}")
        
        # Log individual sensor results
        for sensor, result in results['sensor_results'].items():
            if result['status'] == 'success':
                logger.info(f"  {sensor}: {result['quality']} quality, "
                          f"{result['samples_processed']} samples")
            else:
                logger.error(f"  {sensor}: {result['error']}")
    
    # Create WP1 done marker
    if args.output_dir:
        marker_path = args.output_dir / 'wp1_done.flag'
    else:
        marker_path = Path('aligned_data') / 'wp1_done.flag'
    
    with open(marker_path, 'w') as f:
        f.write(f"WP-1 completed at {pd.Timestamp.now().isoformat()}\n")
    
    logger.info(f"\nWP-1 complete. Done marker created: {marker_path}")
    return 0


def run_wp2(args, config, logger):
    """Execute Work Package 2: Welch PSD core processing."""
    
    # Import WP-2 specific functions
    try:
        from .wp2_process import process_experiment
    except ImportError:
        # If relative import fails, try absolute
        from src.analysis.rpm.wp2_process import process_experiment
    
    # Validate required arguments
    if not args.all and (not args.exp or not args.session):
        logger.error("Either --all or both --exp and --session must be specified")
        return 1
    
    # Get base paths
    base_path = Path(__file__).parent.parent.parent
    output_base = args.output_dir
    
    # Process experiments
    if args.all:
        if not args.session:
            logger.error("--session must be specified with --all")
            return 1
        
        logger.info(f"Processing all experiments for {args.session} session")
        # Get all experiments
        experiments = list_available_experiments()
        exp_list = experiments.get(args.session, [])
        
        if not exp_list:
            logger.error(f"No experiments found for {args.session} session")
            return 1
        
        logger.info(f"Found {len(exp_list)} experiments to process")
        
        # Process each experiment
        all_results = {}
        for exp in exp_list:
            logger.info(f"\nProcessing {exp}...")
            results = process_experiment(
                exp, args.session, config,
                base_path, output_base,
                sensors=args.sensors
            )
            all_results[exp] = results
        
        # Summary
        total_processed = len(all_results)
        successful = sum(1 for r in all_results.values() if r)
        
        logger.info(f"\nWP-2 batch processing complete:")
        logger.info(f"  Experiments processed: {total_processed}")
        logger.info(f"  Successful: {successful}")
        
    else:
        # Process single experiment
        if args.dry_run:
            logger.info("DRY RUN - No processing will be performed")
            logger.info(f"Would process: {args.exp} ({args.session})")
            logger.info(f"Would process sensors: {args.sensors or 'from config'}")
            logger.info(f"Method: {args.method}")
            return 0
        
        results = process_experiment(
            args.exp, args.session, config,
            base_path, output_base,
            sensors=args.sensors
        )
        
        # Log results
        if results:
            logger.info(f"\nProcessing complete for {args.exp}:")
            for sensor_id, path in results.items():
                logger.info(f"  {sensor_id}: {path}")
        else:
            logger.error(f"No results generated for {args.exp}")
    
    # Create completion marker
    marker_path = output_base / "wp2" / "wp2_done.flag"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(f"WP-2 completed at {pd.Timestamp.now()}\n")
    logger.info(f"Created completion marker: {marker_path}")
    
    return 0


def run_wp3(args, config, logger):
    """Execute Work Package 3: STFT + order tracking for transients."""
    
    # Import WP-3 specific functions
    try:
        from .wp3_process import process_experiment
    except ImportError:
        # If relative import fails, try absolute
        from src.analysis.rpm.wp3_process import process_experiment
    
    # Validate required arguments
    if not args.all and (not args.exp or not args.session):
        logger.error("Either --all or both --exp and --session must be specified")
        return 1
    
    # Add WP-3 specific arguments handling
    wp3_args = {}
    
    # SNR threshold override
    if hasattr(args, 'snr_threshold'):
        wp3_args['snr_threshold'] = args.snr_threshold
    
    # Smoothing control
    if hasattr(args, 'no_smoothing'):
        wp3_args['smoothing_enabled'] = not args.no_smoothing
    
    # Edge padding method
    if hasattr(args, 'edge_padding'):
        wp3_args['edge_padding'] = args.edge_padding
    
    # Process experiments
    if args.all:
        if not args.session:
            logger.error("--session must be specified with --all")
            return 1
        
        logger.info(f"Processing all experiments for {args.session} session with STFT")
        # Get all experiments
        experiments = list_available_experiments()
        exp_list = experiments.get(args.session, [])
        
        if not exp_list:
            logger.error(f"No experiments found for {args.session} session")
            return 1
        
        logger.info(f"Found {len(exp_list)} experiments to process")
        
        # Process each experiment
        all_results = {}
        for exp in exp_list:
            logger.info(f"\nProcessing {exp} with STFT...")
            results = process_experiment(
                exp, args.session,
                sensors=args.sensors,
                config_path=args.config,
                generate_plots=args.plot
            )
            all_results[exp] = results
        
        # Summary
        total_processed = len(all_results)
        successful = sum(1 for r in all_results.values() if r)
        
        logger.info(f"\nWP-3 batch processing complete:")
        logger.info(f"  Experiments processed: {total_processed}")
        logger.info(f"  Successful: {successful}")
        
    else:
        # Process single experiment
        if args.dry_run:
            logger.info("DRY RUN - No processing will be performed")
            logger.info(f"Would process: {args.exp} ({args.session})")
            logger.info(f"Would process sensors: {args.sensors or 'from config'}")
            logger.info(f"Method: STFT with early SNR gating")
            logger.info(f"Edge padding: {wp3_args.get('edge_padding', 'mirror')}")
            logger.info(f"Smoothing: {not wp3_args.get('no_smoothing', False)}")
            return 0
        
        results = process_experiment(
            args.exp, args.session,
            sensors=args.sensors,
            config_path=args.config,
            generate_plots=args.plot
        )
        
        # Log results
        if results:
            logger.info(f"\nSTFT processing complete for {args.exp}:")
            for sensor_id, path in results.items():
                logger.info(f"  {sensor_id}: {path}")
        else:
            logger.error(f"No results generated for {args.exp}")
    
    # Create completion marker
    output_base = args.output_dir or Path('results')
    marker_path = output_base / "wp3" / "wp3_done.flag"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(f"WP-3 completed at {pd.Timestamp.now()}\n")
    logger.info(f"Created completion marker: {marker_path}")
    
    return 0


def run_wp4(args, config, logger):
    """Execute Work Package 4: Multi-sensor fusion."""
    
    # Import WP-4 specific functions
    try:
        from .wp4_process import main as process_fusion
    except ImportError:
        # If relative import fails, try absolute
        from src.analysis.rpm.wp4_process import main as process_fusion
    
    # Validate required arguments
    if not args.all and (not args.exp or not args.session):
        logger.error("Either --all or both --exp and --session must be specified")
        return 1
    
    # Update config with CLI overrides
    if args.fusion_strategy:
        if 'wp4' not in config:
            config['wp4'] = {}
        if 'fusion' not in config['wp4']:
            config['wp4']['fusion'] = {}
        config['wp4']['fusion']['strategy'] = args.fusion_strategy
    
    if args.min_sensors:
        config['wp4']['fusion']['min_sensors_required'] = args.min_sensors
    
    if args.interpolation_window:
        if 'interpolation' not in config['wp4']:
            config['wp4']['interpolation'] = {}
        config['wp4']['interpolation']['max_gap_s'] = args.interpolation_window
    
    # Process experiments
    if args.all:
        if not args.session:
            logger.error("--session must be specified with --all")
            return 1
        
        logger.info(f"Processing all experiments for {args.session} session with fusion")
        # Get all experiments
        experiments = list_available_experiments()
        exp_list = experiments.get(args.session, [])
        
        if not exp_list:
            logger.error(f"No experiments found for {args.session} session")
            return 1
        
        logger.info(f"Found {len(exp_list)} experiments to process")
        
        # Process each experiment
        all_results = {}
        failed_count = 0
        for exp in exp_list:
            logger.info(f"\nProcessing fusion for {exp}...")
            try:
                result = process_fusion(
                    exp, args.session, 
                    config_path=args.config,
                    output_dir=args.output_dir,
                    plot=args.plot
                )
                all_results[exp] = result
            except Exception as e:
                logger.error(f"Failed to process {exp}: {e}")
                failed_count += 1
        
        # Summary
        total_processed = len(all_results)
        
        logger.info(f"\nWP-4 batch processing complete:")
        logger.info(f"  Experiments processed: {total_processed}")
        logger.info(f"  Failed: {failed_count}")
        
        # Overall statistics
        if all_results:
            avg_availability = np.mean([r.quality_stats['availability'] 
                                      for r in all_results.values()])
            logger.info(f"  Average availability: {avg_availability:.1f}%")
        
    else:
        # Process single experiment
        if args.dry_run:
            logger.info("DRY RUN - No processing will be performed")
            logger.info(f"Would process: {args.exp} ({args.session})")
            logger.info(f"Fusion strategy: {args.fusion_strategy or 'snr_max'}")
            logger.info(f"Min sensors: {args.min_sensors or 'from config'}")
            logger.info(f"Interpolation window: {args.interpolation_window or 5.0} s")
            return 0
        
        try:
            result = process_fusion(
                args.exp, args.session,
                config_path=args.config,
                output_dir=args.output_dir,
                plot=args.plot
            )
            
            # Log results
            if result:
                logger.info(f"\nFusion complete for {args.exp}:")
                logger.info(f"  Availability: {result.quality_stats['availability']:.1f}%")
                logger.info(f"  Mean SNR: {result.quality_stats['mean_snr_db']:.1f} dB")
                logger.info(f"  Interpolated: {result.quality_stats['interpolated_fraction']*100:.1f}%")
                logger.info(f"  Sensor contributions: {result.sensor_contributions}")
            else:
                logger.error(f"No results generated for {args.exp}")
                return 1
                
        except Exception as e:
            logger.error(f"Error processing {args.exp}: {e}")
            return 1
    
    # Create completion marker
    output_base = args.output_dir or Path('results')
    marker_path = output_base / "wp4" / "wp4_done.flag"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(f"WP-4 completed at {pd.Timestamp.now()}\n")
    logger.info(f"Created completion marker: {marker_path}")
    
    return 0


def run_wp1_validation(args, config, logger):
    """Run WP-1 validation tests including synthetic data test."""
    import numpy as np
    
    logger.info("Running WP-1 validation suite")
    
    # Test 1: Synthetic sine burst test
    if args.include_synthetic:
        logger.info("\nTest 1: Synthetic 25 Hz sine burst")
        
        # Generate synthetic data
        fs = config['fs']
        duration = 10  # seconds
        t = np.arange(0, duration, 1/fs)
        
        # 25 Hz sine wave with amplitude 1 m/s²
        signal = np.sin(2 * np.pi * 25 * t)
        
        # Add white noise for SNR = 30 dB
        noise_power = 10**(-30/10)  # Signal power = 1
        noise = np.sqrt(noise_power) * np.random.randn(len(t))
        noisy_signal = signal + noise
        
        # Process through pipeline
        from .preprocess import high_pass_filter, compute_quality_metrics
        
        # Apply high-pass filter
        filtered = high_pass_filter(noisy_signal, fs, 
                                  config['wp1']['filters']['highpass_cutoff'])
        
        # Compute metrics
        metrics = compute_quality_metrics(filtered)
        
        # Calculate SNR
        signal_power = np.mean(signal**2)
        noise_actual = filtered - signal
        noise_power_actual = np.mean(noise_actual**2)
        snr_db = 10 * np.log10(signal_power / noise_power_actual)
        
        logger.info(f"  Input SNR: 30 dB")
        logger.info(f"  Output SNR: {snr_db:.1f} dB")
        logger.info(f"  Peak-to-RMS: {metrics['peak_to_rms']:.2f}")
        
        if snr_db >= 25:
            logger.info("  ✓ PASSED: SNR ≥ 25 dB")
        else:
            logger.error("  ✗ FAILED: SNR < 25 dB")
    
    # Test 2: Configuration validation
    logger.info("\nTest 2: Configuration validation")
    wp1_config = config.get('wp1', {})
    required_keys = ['sensors', 'filters', 'quality', 'processing', 'output']
    
    missing_keys = [k for k in required_keys if k not in wp1_config]
    if missing_keys:
        logger.error(f"  ✗ FAILED: Missing config keys: {missing_keys}")
    else:
        logger.info("  ✓ PASSED: All required config keys present")
    
    # Test 3: Import validation
    logger.info("\nTest 3: Module imports")
    try:
        from . import io, preprocess, quality, schema, logging_config
        logger.info("  ✓ PASSED: All modules import successfully")
    except ImportError as e:
        logger.error(f"  ✗ FAILED: Import error: {e}")
    
    logger.info("\nValidation complete")
    return 0


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup structured logging
    logger = setup_structured_logging(
        log_file=args.log_file,
        log_level=args.log_level,
        log_format=args.log_format
    )
    
    try:
        # Load configuration
        if not args.config.exists():
            # Check standard locations
            module_dir = Path(__file__).parent
            project_root = module_dir.parent.parent.parent
            
            # Check multiple standard locations
            possible_locations = [
                module_dir / 'rpm_config.yaml',  # In the module directory
                project_root / 'config' / 'processing' / 'rpm_config.yaml',  # Standard config location
                project_root / 'config' / 'rpm_config.yaml',  # Alternative location
            ]
            
            for location in possible_locations:
                if location.exists():
                    args.config = location
                    break
            else:
                raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Route to appropriate work package
        if args.wp == 0:
            logger.info("WP-0: Repository scaffold already complete")
            return 0
        elif args.wp == 1:
            return run_wp1(args, config, logger)
        elif args.wp == 2:
            return run_wp2(args, config, logger)
        elif args.wp == 3:
            return run_wp3(args, config, logger)
        elif args.wp == 4:
            return run_wp4(args, config, logger)
        else:
            logger.error(f"Work package {args.wp} not yet implemented")
            return 1
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())