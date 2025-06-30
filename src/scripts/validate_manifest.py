#!/usr/bin/env python3
"""
Command-line tool for validating experiment manifest files.

Usage:
    python validate_manifest.py [options]
    
Options:
    --manifest PATH     Path to manifest file (default: config/experiments/experiment_manifest.yaml)
    --output PATH       Save report to file (default: print to stdout)
    --format FORMAT     Output format: json or markdown (default: markdown)
    --no-filesystem     Skip filesystem validation checks
    --data-root PATH    Root directory for data (default: /data/raw)
"""

import argparse
import sys
from pathlib import Path

from src.core.validation_report import generate_validation_report_cli


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description='Validate experiment manifest YAML file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'config' / 'experiments' / 'experiment_manifest.yaml',
        help='Path to experiment manifest YAML file'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Save report to file (default: print to stdout)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'markdown'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    
    parser.add_argument(
        '--no-filesystem',
        action='store_true',
        help='Skip filesystem validation checks'
    )
    
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help='Root directory for data (default: uses configured path)'
    )
    
    args = parser.parse_args()
    
    # Check if manifest exists
    if not args.manifest.exists():
        print(f"Error: Manifest file not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)
    
    # Run validation
    try:
        generate_validation_report_cli(
            manifest_path=args.manifest,
            output_path=args.output,
            format=args.format,
            check_filesystem=not args.no_filesystem
        )
    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()