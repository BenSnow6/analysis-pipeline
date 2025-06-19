"""
Command-line interface for RPM estimation.

This module provides the CLI entry point for running RPM estimation
on hovercraft IMU data.
"""

import argparse
import sys
import logging
from pathlib import Path


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='RPM Estimation from IMU Vibration Data',
        epilog='Example: python -m rpm_estimation.cli --exp 026_Engine_rpm_sweep --session afternoon'
    )
    
    # Required arguments
    parser.add_argument('--exp', required=True, 
                       help='Experiment name (e.g., 026_Engine_rpm_sweep)')
    parser.add_argument('--session', choices=['morning', 'afternoon'],
                       required=True, help='Data collection session')
    
    # Method selection
    parser.add_argument('--method', choices=['welch', 'stft', 'both'],
                       default='welch', help='Estimation method (default: welch)')
    
    # Configuration
    parser.add_argument('--config', default='rpm_config.yaml',
                       type=Path, help='Path to configuration file')
    
    # Output options
    parser.add_argument('--output-dir', default='results/',
                       type=Path, help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    # Sensor selection
    parser.add_argument('--sensors', nargs='+',
                       default=['Sensor_3', 'Sensor_4', 'Sensor_wb'],
                       help='Sensors to process (default: validated sensors)')
    
    # Processing options
    parser.add_argument('--no-fusion', action='store_true',
                       help='Skip multi-sensor fusion')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate processing results')
    
    # Debugging
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without running')
    
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
        # If default config doesn't exist in current dir, check module dir
        module_dir = Path(__file__).parent
        default_config = module_dir / 'rpm_config.yaml'
        if default_config.exists():
            args.config = default_config
        else:
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    return args


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate arguments
        args = validate_args(args)
        
        # Log configuration
        logger.info(f"RPM estimation for {args.exp} ({args.session} session)")
        logger.info(f"Method: {args.method}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Sensors: {', '.join(args.sensors)}")
        
        if args.dry_run:
            logger.info("DRY RUN - No processing will be performed")
            logger.info(f"Would process data from: {args.exp}")
            logger.info(f"Would save results to: {args.output_dir}")
            return 0
        
        # Placeholder for future implementation
        logger.info("WP-0 scaffold complete - ready for WP-1 implementation")
        logger.info("Run with --help to see all available options")
        
        # TODO: Implement actual processing in WP-1 and beyond
        # - Load data using io.py
        # - Preprocess using preprocess.py
        # - Compute RPM using spectral.py
        # - Fuse sensors using fusion.py
        # - Save results
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())