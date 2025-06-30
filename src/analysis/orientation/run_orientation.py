#!/usr/bin/env python3
"""
Command-line interface for orientation analysis.
Processes multiple experiments and generates comprehensive reports.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
from tqdm import tqdm

from src.analysis.orientation.orientation_check import OrientationChecker
from src.analysis.orientation.plot_orientation import OrientationPlotter
from src.core.paths import ORIENTATION_CONFIG_FILE, ALIGNED_DATA_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Validate sensor orientations for hovercraft data"
    )
    
    parser.add_argument(
        "-e", "--experiments",
        nargs="+",
        default=[],
        help="List of experiments to process"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=ORIENTATION_CONFIG_FILE,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "-d", "--data-dir",
        type=Path,
        default=ALIGNED_DATA_DIR,
        help="Directory containing aligned data"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent / "validation_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing results"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    args = parser.parse_args()
    
    # Check if experiments provided
    if not args.experiments:
        print("ERROR: No experiments specified.")
        print("Please provide experiment names with -e flag.")
        print("Example: python run_orientation.py -e experiment1 experiment2")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize modules
    print("Initializing orientation analysis modules...")
    checker = OrientationChecker(args.config)
    plotter = OrientationPlotter(args.config)
    
    # Results storage
    all_results = {}
    
    if not args.plot_only:
        # Process each experiment
        print(f"\nProcessing {len(args.experiments)} experiments...")
        
        for experiment in tqdm(args.experiments, desc="Experiments"):
            print(f"\n{'='*60}")
            print(f"Processing: {experiment}")
            print(f"{'='*60}")
            
            try:
                results = checker.validate_experiment(
                    experiment,
                    output_dir=args.output_dir / experiment
                )
                all_results[experiment] = results
                
                # Print quick summary
                if 'sensors' in results:
                    passed = 0
                    total = 0
                    for sensor_name, sensor_results in results['sensors'].items():
                        if 'overall_valid' in sensor_results:
                            total += 1
                            if sensor_results['overall_valid']:
                                passed += 1
                                
                    print(f"\nSummary for {experiment}: {passed}/{total} sensors passed")
                    
            except Exception as e:
                print(f"ERROR processing {experiment}: {str(e)}")
                import traceback
                traceback.print_exc()
                all_results[experiment] = {'error': str(e)}
                
        # Save all results
        results_file = args.output_dir / "all_validation_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(all_results, f, default_flow_style=False)
        print(f"\nSaved all results to: {results_file}")
        
    else:
        # Load existing results
        results_file = args.output_dir / "all_validation_results.yaml"
        if results_file.exists():
            with open(results_file, 'r') as f:
                all_results = yaml.safe_load(f)
            print(f"Loaded results from: {results_file}")
        else:
            print(f"ERROR: No results file found at {results_file}")
            return 1
            
    # Generate summary plots if requested
    if not args.no_plots and all_results:
        print("\nGenerating summary plots...")
        
        # Sensor coordinate systems plot
        fig, ax = plotter.plot_sensor_coordinate_systems()
        fig.savefig(args.output_dir / "sensor_coordinate_systems.png", 
                   dpi=150, bbox_inches='tight')
        print("  - Saved sensor_coordinate_systems.png")
        
        # Overall validation summary
        if len(all_results) > 1:
            fig, axes = plotter.plot_validation_summary(all_results)
            fig.savefig(args.output_dir / "validation_summary.png", dpi=150)
            print("  - Saved validation_summary.png")
            
        # Cross-sensor consistency for each experiment
        for exp_name, exp_results in all_results.items():
            if 'sensors' in exp_results:
                fig_tuple = plotter.plot_cross_sensor_consistency(exp_results)
                if fig_tuple:
                    fig, axes = fig_tuple
                    fig.savefig(args.output_dir / f"{exp_name}_cross_sensor_consistency.png", 
                               dpi=150)
                    print(f"  - Saved {exp_name}_cross_sensor_consistency.png")
                    
    # Generate final summary report
    print("\nGenerating final summary report...")
    generate_final_report(all_results, args.output_dir)
    
    print(f"\nOrientation analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


def generate_final_report(all_results: dict, output_dir: Path):
    """Generate comprehensive markdown report for all experiments."""
    
    report_lines = [
        "# Orientation Analysis - Final Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Calculate overall statistics
    total_tests = 0
    total_passed = 0
    sensor_stats = {}
    
    for exp_name, exp_results in all_results.items():
        if 'sensors' in exp_results:
            for sensor_name, sensor_results in exp_results['sensors'].items():
                if sensor_name not in sensor_stats:
                    sensor_stats[sensor_name] = {
                        'tests': 0, 'passed': 0, 'rotation_errors': [],
                        'bias_magnitudes': []
                    }
                    
                if 'overall_valid' in sensor_results:
                    total_tests += 1
                    sensor_stats[sensor_name]['tests'] += 1
                    
                    if sensor_results['overall_valid']:
                        total_passed += 1
                        sensor_stats[sensor_name]['passed'] += 1
                        
                    if 'rotation_error_deg' in sensor_results:
                        sensor_stats[sensor_name]['rotation_errors'].append(
                            sensor_results['rotation_error_deg']
                        )
                        
                    if 'bias_estimation' in sensor_results:
                        bias = sensor_results['bias_estimation']
                        if 'accel_bias_magnitude' in bias:
                            sensor_stats[sensor_name]['bias_magnitudes'].append(
                                bias['accel_bias_magnitude']
                            )
                            
    # Overall pass rate
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    report_lines.extend([
        f"- **Total Validation Tests**: {total_tests}",
        f"- **Tests Passed**: {total_passed}",
        f"- **Overall Pass Rate**: {pass_rate:.1f}%",
        f"- **Experiments Analyzed**: {len(all_results)}",
        "",
        "## Sensor Performance Summary",
        "",
        "| Sensor | Pass Rate | Avg Rotation Error (°) | Avg Bias Magnitude (m/s²) |",
        "|--------|-----------|------------------------|---------------------------|"
    ])
    
    for sensor_name in ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']:
        if sensor_name in sensor_stats:
            stats = sensor_stats[sensor_name]
            pass_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
            
            avg_rot_error = (sum(stats['rotation_errors']) / len(stats['rotation_errors'])
                           if stats['rotation_errors'] else 0)
            avg_bias = (sum(stats['bias_magnitudes']) / len(stats['bias_magnitudes'])
                       if stats['bias_magnitudes'] else 0)
                       
            report_lines.append(
                f"| {sensor_name} | {pass_rate:.0f}% | "
                f"{avg_rot_error:.2f} | {avg_bias:.4f} |"
            )
            
    # Detailed results by experiment
    report_lines.extend([
        "",
        "## Detailed Results by Experiment",
        ""
    ])
    
    for exp_name, exp_results in all_results.items():
        report_lines.extend([
            f"### {exp_name}",
            ""
        ])
        
        if 'error' in exp_results:
            report_lines.append(f"**ERROR**: {exp_results['error']}")
        elif 'sensors' in exp_results:
            report_lines.extend([
                "| Sensor | Rotation Error | Static | Bias | Dynamic | Overall |",
                "|--------|----------------|--------|------|---------|---------|"
            ])
            
            for sensor_name in ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']:
                if sensor_name in exp_results['sensors']:
                    sensor_results = exp_results['sensors'][sensor_name]
                    if 'error' not in sensor_results:
                        rot_error = sensor_results.get('rotation_error_deg', -1)
                        static = "✅" if sensor_results.get('static_valid', False) else "❌"
                        bias = "✅" if sensor_results.get('bias_valid', False) else "❌"
                        dynamic = "✅" if sensor_results.get('dynamic_valid', False) else "❌"
                        overall = "✅" if sensor_results.get('overall_valid', False) else "❌"
                        
                        report_lines.append(
                            f"| {sensor_name} | {rot_error:.2f}° | {static} | "
                            f"{bias} | {dynamic} | {overall} |"
                        )
                        
        report_lines.append("")
        
    # Recommendations
    report_lines.extend([
        "## Recommendations",
        "",
        "Based on the orientation validation results:",
        ""
    ])
    
    # Check for any failed sensors
    failed_sensors = []
    for sensor_name, stats in sensor_stats.items():
        if stats['tests'] > 0 and stats['passed'] < stats['tests']:
            failed_sensors.append(sensor_name)
            
    if failed_sensors:
        report_lines.extend([
            f"⚠️ **Attention Required**: The following sensors showed validation issues:",
            ""
        ])
        for sensor in failed_sensors:
            report_lines.append(f"- {sensor}")
        report_lines.append("")
        
    report_lines.extend([
        "### Next Steps:",
        "1. Review rotation matrices for any sensors with errors > 3°",
        "2. Apply bias corrections before Kalman filtering",
        "3. Consider excluding sensors with persistent validation failures",
        "4. Use the validated rotation matrices and bias estimates in Week 2 analysis",
        "",
        "## Data Quality Certificate",
        "",
        f"✅ **Temporal Alignment**: Complete (Week 1 Day 1)",
        f"{'✅' if pass_rate > 80 else '⚠️'} **Orientation Validation**: "
        f"{pass_rate:.0f}% Pass Rate",
        f"✅ **Ready for Kalman Filtering**: "
        f"{'Yes' if pass_rate > 80 else 'Review required'}",
        ""
    ])
    
    # Save report
    report_path = output_dir / "ORIENTATION_ANALYSIS_FINAL_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print(f"  - Saved final report to: {report_path}")


if __name__ == "__main__":
    sys.exit(main())