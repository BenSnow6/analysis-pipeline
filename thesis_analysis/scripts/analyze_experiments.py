#!/usr/bin/env python3
"""
Main analysis script for processing all hovercraft experiments.
This script generates plots, statistics, and prepares data for simulator comparison.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Optional

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from plotting.experiment_plots import ExperimentPlotter, process_all_experiments


class ExperimentAnalyzer:
    """Main class for analyzing hovercraft experiments."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.experiments_dir = self.base_path / "02_Evaluation_Experiments"
        self.results_dir = self.base_path / "analysis_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load experiment catalog
        self.experiments = self._load_experiment_catalog()
        
    def _load_experiment_catalog(self) -> Dict:
        """Load and parse the experiment catalog."""
        catalog = {
            '1a_1_Minimum_Radius_Turn': {
                'description': 'Minimum turning radius capabilities',
                'morning': ['015_Skirt_shift_turns'],
                'afternoon': ['007_Fast_stbd_turn_1', '009_Fast_port_turn_1', 
                            '011_Static_stbd_1', '012_Static_port_1',
                            '013_Static_port_2', '014_Static_stbd_2']
            },
            '1a_2_Rate_of_Turn_vs_Nosewheel_Steering_Angle': {
                'description': 'Relationship between steering input and turn rate',
                'afternoon': ['021_Quarter_turn_port', '022_Quarter_turn_stbd',
                            '023_Eigth_turn_port', '024_Eigth_turn_stbd']
            },
            '1b_1_Ground_Acceleration_Time_and_Distance': {
                'description': 'Acceleration performance and distance requirements',
                'morning': ['007_Downwind_max_speed_1', '009_Downwind_max_speed_2',
                          '010_Downwind_max_speed_3'],
                'afternoon': ['016_Straight_cruise_1', '018_Straight_cruise_2',
                            '020_Straight_cruise_3']
            },
            '1b_4_Normal_Take_off': {
                'description': 'Normal take-off procedures and performance',
                'morning': ['006_Departure', '013_Yaw_speed_3'],
                'afternoon': ['026_Engine_rpm_sweep']
            },
            '1c_1_Normal_Climb_All_Engines_Operating': {
                'description': 'Climb performance with all engines operational',
                'morning': ['014_Floating_on_sea_and_takeoff', '016_Plough_in']
            },
            '1d_1_Level_Flight_Acceleration': {
                'description': 'Acceleration in level flight conditions',
                'morning': ['007_Downwind_max_speed_1', '009_Downwind_max_speed_2',
                          '010_Downwind_max_speed_3']
            },
            '1d_2_Level_Flight_Deceleration': {
                'description': 'Deceleration in level flight conditions',
                'morning': ['013_Yaw_speed_3']
            }
        }
        return catalog
    
    def analyze_experiment(self, category: str, time_slot: str, experiment: str) -> Dict:
        """Analyze a single experiment and return statistics."""
        exp_path = f"{category}/{time_slot}/{experiment}"
        full_path = self.experiments_dir / exp_path
        
        if not full_path.exists():
            print(f"Warning: Experiment path does not exist: {full_path}")
            return {}
        
        stats = {
            'experiment': experiment,
            'category': category,
            'time_slot': time_slot,
            'path': exp_path
        }
        
        try:
            # Load GPS data for basic statistics
            gps_file = full_path / "GPS" / f"GPS_{experiment}.csv"
            if gps_file.exists():
                gps_df = pd.read_csv(gps_file)
                stats['gps_stats'] = {
                    'duration_s': gps_df['time_from_sync'].max() - gps_df['time_from_sync'].min(),
                    'max_speed_kph': gps_df['SpeedKPH'].max(),
                    'avg_speed_kph': gps_df['SpeedKPH'].mean(),
                    'distance_m': gps_df['Dst'].sum() if 'Dst' in gps_df.columns else None,
                    'num_satellites': gps_df['Sats'].mean() if 'Sats' in gps_df.columns else None
                }
                
                # Calculate turn statistics for turn experiments
                if 'turn' in experiment.lower() or 'turn' in category.lower():
                    bearing_diff = np.diff(gps_df['Bearing'].values)
                    bearing_diff = np.where(bearing_diff > 180, bearing_diff - 360, bearing_diff)
                    bearing_diff = np.where(bearing_diff < -180, bearing_diff + 360, bearing_diff)
                    time_diff = np.diff(gps_df['time_from_sync'].values)
                    turn_rates = bearing_diff / time_diff
                    
                    stats['turn_stats'] = {
                        'max_turn_rate_deg_s': np.abs(turn_rates).max(),
                        'avg_turn_rate_deg_s': np.abs(turn_rates).mean(),
                        'total_heading_change_deg': np.abs(bearing_diff).sum()
                    }
                
                # Calculate acceleration statistics
                speed_ms = gps_df['SpeedKPH'].values * 0.27778
                time_diff = np.diff(gps_df['time_from_sync'].values)
                accelerations = np.diff(speed_ms) / time_diff
                
                stats['acceleration_stats'] = {
                    'max_acceleration_ms2': accelerations.max(),
                    'max_deceleration_ms2': abs(accelerations.min()),
                    'avg_acceleration_ms2': accelerations[accelerations > 0].mean() if any(accelerations > 0) else 0
                }
                
        except Exception as e:
            print(f"Error analyzing {exp_path}: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report of all experiments."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'experiments': []
        }
        
        # Analyze each experiment
        for category, info in self.experiments.items():
            for time_slot in ['morning', 'afternoon']:
                if time_slot in info:
                    for experiment in info[time_slot]:
                        stats = self.analyze_experiment(category, time_slot, experiment)
                        if stats:
                            stats['description'] = info['description']
                            report['experiments'].append(stats)
        
        # Save report
        report_file = self.results_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown summary
        self._generate_markdown_summary(report)
        
        return report
    
    def _generate_markdown_summary(self, report: Dict):
        """Generate a markdown summary of the analysis."""
        md_content = f"""# Hovercraft Experiment Analysis Summary

Generated: {report['generated_at']}

## Overview

Total experiments analyzed: {len(report['experiments'])}

## Experiment Categories

"""
        
        # Group by category
        categories = {}
        for exp in report['experiments']:
            cat = exp['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(exp)
        
        for category, experiments in categories.items():
            md_content += f"### {category.replace('_', ' ')}\n\n"
            md_content += f"**Description:** {experiments[0]['description']}\n\n"
            md_content += "| Experiment | Time | Duration (s) | Max Speed (km/h) | "
            
            # Add turn-specific columns if applicable
            if 'turn' in category.lower():
                md_content += "Max Turn Rate (°/s) | Total Turn (°) |\n"
                md_content += "|------------|------|--------------|------------------|---------------------|----------------|\n"
            else:
                md_content += "Max Accel (m/s²) |\n"
                md_content += "|------------|------|--------------|------------------|------------------|\n"
            
            for exp in experiments:
                if 'gps_stats' in exp:
                    gs = exp['gps_stats']
                    row = f"| {exp['experiment']} | {exp['time_slot']} | "
                    row += f"{gs['duration_s']:.1f} | {gs['max_speed_kph']:.1f} | "
                    
                    if 'turn_stats' in exp:
                        ts = exp['turn_stats']
                        row += f"{ts['max_turn_rate_deg_s']:.1f} | {ts['total_heading_change_deg']:.1f} |"
                    elif 'acceleration_stats' in exp:
                        as_ = exp['acceleration_stats']
                        row += f"{as_['max_acceleration_ms2']:.2f} |"
                    
                    md_content += row + "\n"
            
            md_content += "\n"
        
        # Save markdown file
        md_file = self.results_dir / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(md_file, 'w') as f:
            f.write(md_content)
        
        print(f"Summary report saved to: {md_file}")
    
    def process_all_experiments(self, generate_plots=True):
        """Process all experiments with plots and analysis."""
        print("Starting comprehensive experiment analysis...")
        
        # Generate plots if requested
        if generate_plots:
            print("\nGenerating plots for all experiments...")
            process_all_experiments(self.experiments_dir)
        
        # Generate summary report
        print("\nGenerating analysis summary...")
        report = self.generate_summary_report()
        
        print(f"\nAnalysis complete! Results saved to: {self.results_dir}")
        
        return report
    
    def prepare_simulator_comparison_data(self):
        """Prepare data for simulator comparison."""
        comparison_dir = self.results_dir / "simulator_comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        # Create standardized data format for each experiment
        for category, info in self.experiments.items():
            for time_slot in ['morning', 'afternoon']:
                if time_slot in info:
                    for experiment in info[time_slot]:
                        self._export_experiment_data(category, time_slot, experiment, comparison_dir)
        
        print(f"Simulator comparison data prepared in: {comparison_dir}")
    
    def _export_experiment_data(self, category: str, time_slot: str, 
                               experiment: str, output_dir: Path):
        """Export experiment data in standardized format for simulator comparison."""
        exp_path = self.experiments_dir / category / time_slot / experiment
        
        if not exp_path.exists():
            return
        
        # Create output directory for this experiment
        exp_output = output_dir / f"{category}_{time_slot}_{experiment}"
        exp_output.mkdir(exist_ok=True)
        
        # Export GPS data
        gps_file = exp_path / "GPS" / f"GPS_{experiment}.csv"
        if gps_file.exists():
            gps_df = pd.read_csv(gps_file)
            # Select relevant columns
            gps_export = gps_df[['time_from_sync', 'Lat', 'Lng', 'Alt', 
                               'SpeedKPH', 'Bearing']].copy()
            gps_export.to_csv(exp_output / "gps_data.csv", index=False)
        
        # Export IMU data (primary sensor)
        for data_type in ['accel', 'gyro', 'mag', 'angle']:
            imu_file = exp_path / "IMU" / "Sensor_3" / f"{data_type}_{experiment}.csv"
            if imu_file.exists():
                imu_df = pd.read_csv(imu_file)
                imu_export = imu_df[['time_from_sync', 'x', 'y', 'z']].copy()
                imu_export.to_csv(exp_output / f"imu_{data_type}.csv", index=False)
        
        # Create experiment metadata
        metadata = {
            'experiment': experiment,
            'category': category,
            'time_slot': time_slot,
            'description': self.experiments[category]['description'],
            'data_files': {
                'gps': 'gps_data.csv',
                'imu_accel': 'imu_accel.csv',
                'imu_gyro': 'imu_gyro.csv',
                'imu_mag': 'imu_mag.csv',
                'imu_angle': 'imu_angle.csv'
            }
        }
        
        with open(exp_output / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(description='Analyze hovercraft experiment data')
    parser.add_argument('--base-path', type=str, 
                       default=str(Path(__file__).parent.parent.parent),
                       help='Base path for the analysis pipeline')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--simulator-prep', action='store_true',
                       help='Prepare data for simulator comparison')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ExperimentAnalyzer(args.base_path)
    
    # Process all experiments
    analyzer.process_all_experiments(generate_plots=not args.no_plots)
    
    # Prepare simulator comparison data if requested
    if args.simulator_prep:
        analyzer.prepare_simulator_comparison_data()
    
    print("\nAnalysis pipeline completed successfully!")


if __name__ == "__main__":
    main()