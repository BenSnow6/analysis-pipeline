#!/usr/bin/env python3
"""
Main entry point for timestamp analysis tool.

This module provides a command-line interface for analyzing timestamp
consistency in hovercraft sensor data.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import warnings

# Import our modules
from . import data_loader
from . import timestamp_analyzer
from . import visualizer
from . import report_generator


def print_summary(results: Dict[str, timestamp_analyzer.TimestampAnalysisResult],
                 verbose: bool = False) -> None:
    """Print analysis summary to console."""
    print("\n" + "="*80)
    print("TIMESTAMP ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    num_sensors = len(results)
    num_pass = sum(1 for r in results.values() if r.within_spec)
    
    print(f"\nSensors analyzed: {num_sensors}")
    print(f"Sensors within spec: {num_pass}/{num_sensors}")
    
    if num_pass == num_sensors:
        print("\n✓ ALL SENSORS PASS")
    else:
        print(f"\n✗ {num_sensors - num_pass} SENSORS FAIL")
    
    # Per-sensor summary
    print("\n" + "-"*80)
    print(f"{'Sensor':<15} {'Status':<8} {'Rate (Hz)':<12} {'Jitter (ms)':<15} {'Gaps':<8} {'Issues'}")
    print("-"*80)
    
    for sensor_name in sorted(results.keys()):
        result = results[sensor_name]
        status = "✓ PASS" if result.within_spec else "✗ FAIL"
        rate_str = f"{result.actual_rate_hz:.1f} ({result.expected_rate_hz:.0f})"
        jitter_str = f"{result.mean_jitter_ms:.1f} < {result.jitter_threshold_ms:.0f}"
        
        issues_str = ""
        if not result.within_spec:
            if result.issues:
                issues_str = result.issues[0][:30] + "..." if len(result.issues[0]) > 30 else result.issues[0]
        
        print(f"{sensor_name:<15} {status:<8} {rate_str:<12} {jitter_str:<15} {result.num_gaps:<8} {issues_str}")
    
    # Detailed issues if verbose
    if verbose:
        print("\n" + "-"*80)
        print("DETAILED ISSUES:")
        for sensor_name in sorted(results.keys()):
            result = results[sensor_name]
            if not result.within_spec and result.issues:
                print(f"\n{sensor_name}:")
                for issue in result.issues:
                    print(f"  - {issue}")
    
    print("="*80 + "\n")


def analyze_single_experiment(experiment_path: str, 
                            specs: Dict[str, Any],
                            output_dir: Optional[Path] = None,
                            plot: bool = True,
                            verbose: bool = False) -> Dict[str, timestamp_analyzer.TimestampAnalysisResult]:
    """
    Analyze a single experiment.
    
    Args:
        experiment_path: Path to experiment directory
        specs: Sensor specifications
        output_dir: Directory for output files
        plot: Whether to generate plots
        verbose: Verbose output
        
    Returns:
        Dictionary of analysis results
    """
    experiment_name = Path(experiment_path).name
    
    print(f"\nAnalyzing experiment: {experiment_name}")
    print(f"Path: {experiment_path}")
    
    # Load data
    print("Loading sensor data...")
    sensor_data = data_loader.load_experiment_data(experiment_path, specs)
    
    if not sensor_data:
        print("WARNING: No sensor data found!")
        return {}
    
    print(f"Found {len(sensor_data)} sensors with data")
    
    # Analyze timestamps
    print("Analyzing timestamps...")
    results = timestamp_analyzer.analyze_experiment(sensor_data, specs)
    
    # Print summary
    print_summary(results, verbose)
    
    # Generate outputs if requested
    if output_dir:
        output_dir = Path(output_dir)
        exp_output_dir = output_dir / experiment_name.replace("/", "_")
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        if plot:
            print(f"Generating plots in {exp_output_dir}...")
            
            # Individual sensor plots
            for sensor_name, result in results.items():
                visualizer.create_sensor_report_plots(result, exp_output_dir)
            
            # Summary plot
            visualizer.create_experiment_summary_plots(
                results, experiment_name, exp_output_dir
            )
        
        # Generate HTML report
        print("Generating HTML report...")
        alignment_info = timestamp_analyzer.compare_sensor_alignment(results)
        html_content = report_generator.generate_html_report(
            results, experiment_name, alignment_info, include_plots=plot
        )
        
        report_path = exp_output_dir / "timestamp_analysis_report.html"
        report_generator.save_html_report(html_content, report_path)
        print(f"Report saved to: {report_path}")
        
        # Generate CSV summary
        csv_path = exp_output_dir / "timestamp_analysis_summary.csv"
        report_generator.generate_summary_csv(results, csv_path)
        print(f"CSV summary saved to: {csv_path}")
        
        # Save raw results as JSON
        json_path = exp_output_dir / "timestamp_analysis_results.json"
        save_results_json(results, json_path)
        print(f"Raw results saved to: {json_path}")
    
    return results


def save_results_json(results: Dict[str, timestamp_analyzer.TimestampAnalysisResult],
                     output_path: Path) -> None:
    """Save analysis results as JSON."""
    json_data = {}
    
    for sensor_name, result in results.items():
        json_data[sensor_name] = {
            'within_spec': result.within_spec,
            'num_samples': int(result.num_samples),
            'duration_seconds': float(result.duration_seconds),
            'expected_rate_hz': float(result.expected_rate_hz),
            'actual_rate_hz': float(result.actual_rate_hz),
            'rate_deviation_percent': float(result.rate_deviation_percent),
            'mean_jitter_ms': float(result.mean_jitter_ms),
            'max_jitter_ms': float(result.max_jitter_ms),
            'jitter_threshold_ms': float(result.jitter_threshold_ms),
            'jitter_violations': int(result.jitter_violations),
            'num_gaps': int(result.num_gaps),
            'gap_threshold_ms': float(result.gap_threshold_ms),
            'issues': result.issues,
            'gaps': result.gaps
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)


def auto_detect_rates(experiment_path: str, output_path: Path) -> None:
    """
    Auto-detect sampling rates and generate suggested spec file.
    
    Args:
        experiment_path: Path to experiment directory
        output_path: Path to save suggested specs
    """
    print(f"\nAuto-detecting sampling rates for: {experiment_path}")
    
    # Load data with default specs
    default_specs = data_loader.get_default_specs()
    sensor_data = data_loader.load_experiment_data(experiment_path, default_specs)
    
    if not sensor_data:
        print("WARNING: No sensor data found!")
        return
    
    # Analyze each sensor to determine actual rate
    detected_specs = {
        'sensors': {},
        'default': default_specs['sensors']['default'],
        'analysis': default_specs['analysis']
    }
    
    for sensor_name, df in sensor_data.items():
        if 'time_from_sync' not in df.columns or len(df) < 2:
            continue
        
        timestamps = df['time_from_sync'].values
        duration = timestamps[-1] - timestamps[0]
        num_intervals = len(timestamps) - 1
        
        if duration > 0:
            actual_rate = num_intervals / duration
            
            # Round to nearest common rate
            common_rates = [1, 10, 20, 50, 100, 200, 250, 500, 1000]
            closest_rate = min(common_rates, key=lambda x: abs(x - actual_rate))
            
            # Determine appropriate jitter threshold
            if closest_rate <= 1:
                jitter_threshold = 100
            elif closest_rate <= 10:
                jitter_threshold = 50
            else:
                jitter_threshold = 20
            
            detected_specs['sensors'][sensor_name] = {
                'expected_rate_hz': closest_rate,
                'jitter_threshold_ms': jitter_threshold,
                'gap_threshold_factor': 10.0,
                '_detected_rate': round(actual_rate, 2),
                '_num_samples': len(timestamps)
            }
            
            print(f"{sensor_name}: Detected {actual_rate:.1f}Hz, suggesting {closest_rate}Hz")
    
    # Save suggested specs
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(detected_specs, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nSuggested specs saved to: {output_path}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze timestamp consistency in hovercraft sensor data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single experiment
  python -m hovercraft_analysis.analysis.timestamp --experiment "1a_1_Minimum_Radius_Turn/afternoon/007_Fast_stbd_turn_1"
  
  # Analyze all experiments
  python -m hovercraft_analysis.analysis.timestamp --all
  
  # Auto-detect sampling rates
  python -m hovercraft_analysis.analysis.timestamp --experiment "path/to/experiment" --update-spec
  
  # Use custom sensor specs
  python -m hovercraft_analysis.analysis.timestamp --experiment "path/to/experiment" --spec custom_specs.yaml
        """
    )
    
    parser.add_argument('--experiment', '-e', type=str,
                       help='Path to specific experiment (relative to data repo)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Analyze all experiments')
    parser.add_argument('--spec', '-s', type=str,
                       help='Path to custom sensor_specs.yaml file')
    parser.add_argument('--output', '-o', type=str, default='timestamp_analysis_output',
                       help='Output directory for reports and plots (default: timestamp_analysis_output)')
    parser.add_argument('--plot', '-p', action='store_true', default=True,
                       help='Generate diagnostic plots (default: True)')
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                       help='Skip plot generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--update-spec', action='store_true',
                       help='Auto-detect sampling rates and generate suggested spec file')
    parser.add_argument('--data-path', type=str,
                       help='Path to data repository (overrides default)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.experiment and not args.all:
        parser.error("Either --experiment or --all must be specified")
    
    if args.experiment and args.all:
        parser.error("Cannot specify both --experiment and --all")
    
    # Load sensor specs
    specs = data_loader.load_sensor_specs(args.spec)
    
    # Set data path if provided
    if args.data_path:
        data_loader.DATA_REPO_PATH = args.data_path
    
    # Get available experiments
    experiments = data_loader.get_available_experiments(args.data_path)
    
    if not experiments:
        print("ERROR: No experiments found in data repository!")
        sys.exit(1)
    
    print(f"Found {len(experiments)} experiments in repository")
    
    # Handle single experiment
    if args.experiment:
        if args.experiment not in experiments:
            print(f"ERROR: Experiment '{args.experiment}' not found!")
            print("\nAvailable experiments:")
            for exp_name in sorted(experiments.keys())[:10]:
                print(f"  - {exp_name}")
            if len(experiments) > 10:
                print(f"  ... and {len(experiments) - 10} more")
            sys.exit(1)
        
        experiment_path = experiments[args.experiment]
        
        # Auto-detect mode
        if args.update_spec:
            output_path = Path(args.output) / "suggested_sensor_specs.yaml"
            auto_detect_rates(experiment_path, output_path)
        else:
            # Normal analysis
            analyze_single_experiment(
                experiment_path,
                specs,
                Path(args.output) if args.output else None,
                args.plot,
                args.verbose
            )
    
    # Handle all experiments
    else:
        output_dir = Path(args.output) if args.output else None
        all_results = {}
        failed_experiments = []
        
        for exp_name, exp_path in experiments.items():
            try:
                print(f"\n{'='*80}")
                results = analyze_single_experiment(
                    exp_path,
                    specs,
                    output_dir,
                    args.plot,
                    args.verbose
                )
                all_results[exp_name] = results
            except Exception as e:
                print(f"ERROR analyzing {exp_name}: {str(e)}")
                failed_experiments.append(exp_name)
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        # Summary of all experiments
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Experiments analyzed: {len(all_results)}")
        print(f"Experiments failed: {len(failed_experiments)}")
        
        if output_dir:
            # Save overall summary
            summary_path = output_dir / "all_experiments_summary.json"
            overall_summary = {
                'total_experiments': len(all_results),
                'failed_experiments': failed_experiments,
                'experiment_summaries': {}
            }
            
            for exp_name, results in all_results.items():
                exp_summary = {
                    'sensors_analyzed': len(results),
                    'sensors_passing': sum(1 for r in results.values() if r.within_spec),
                    'overall_pass': all(r.within_spec for r in results.values())
                }
                overall_summary['experiment_summaries'][exp_name] = exp_summary
            
            with open(summary_path, 'w') as f:
                json.dump(overall_summary, f, indent=2)
            
            print(f"\nOverall summary saved to: {summary_path}")


if __name__ == "__main__":
    # Suppress matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    main()