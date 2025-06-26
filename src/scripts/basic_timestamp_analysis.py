#!/usr/bin/env python3
"""
Basic timestamp analysis using only Python standard library.
Provides essential timing analysis without external dependencies.
"""

import csv
import os
import json
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class BasicTimestampAnalyzer:
    def __init__(self):
        self.results = {}
        self.sensor_specs = {
            'gps': {'expected_rate': 1, 'jitter_threshold_ms': 100},
            'default': {'expected_rate': 200, 'jitter_threshold_ms': 20}
        }
    
    def analyze_csv_timestamps(self, filepath, sensor_type='default'):
        """Analyze timestamps from a CSV file."""
        timestamps = []
        
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try different timestamp columns
                    if 'time_from_sync' in row:
                        try:
                            timestamps.append(float(row['time_from_sync']))
                        except:
                            pass
                    elif 't' in row:
                        try:
                            timestamps.append(float(row['t']))
                        except:
                            pass
        except Exception as e:
            return {'error': str(e)}
        
        if len(timestamps) < 2:
            return {'error': 'Insufficient data'}
        
        # Sort timestamps
        timestamps.sort()
        
        # Calculate intervals (in milliseconds)
        intervals = [(timestamps[i+1] - timestamps[i]) * 1000 
                    for i in range(len(timestamps)-1)]
        
        # Get expected values
        spec = self.sensor_specs.get(sensor_type, self.sensor_specs['default'])
        expected_interval_ms = 1000.0 / spec['expected_rate']
        
        # Calculate statistics
        mean_interval = statistics.mean(intervals)
        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
        min_interval = min(intervals)
        max_interval = max(intervals)
        
        # Calculate actual rate
        duration = timestamps[-1] - timestamps[0]
        actual_rate = (len(timestamps) - 1) / duration if duration > 0 else 0
        
        # Calculate jitter (deviation from expected)
        jitter_values = [abs(interval - expected_interval_ms) for interval in intervals]
        mean_jitter = statistics.mean(jitter_values)
        max_jitter = max(jitter_values)
        
        # Count violations
        violations = sum(1 for j in jitter_values if j > spec['jitter_threshold_ms'])
        
        # Detect gaps (intervals > 10x expected)
        gap_threshold = expected_interval_ms * 10
        gaps = [(i, intervals[i]) for i in range(len(intervals)) 
                if intervals[i] > gap_threshold]
        
        return {
            'num_samples': len(timestamps),
            'duration_s': duration,
            'expected_rate_hz': spec['expected_rate'],
            'actual_rate_hz': round(actual_rate, 2),
            'mean_interval_ms': round(mean_interval, 2),
            'std_interval_ms': round(std_interval, 2),
            'min_interval_ms': round(min_interval, 2),
            'max_interval_ms': round(max_interval, 2),
            'mean_jitter_ms': round(mean_jitter, 2),
            'max_jitter_ms': round(max_jitter, 2),
            'jitter_threshold_ms': spec['jitter_threshold_ms'],
            'jitter_violations': violations,
            'num_gaps': len(gaps),
            'within_spec': violations == 0 and len(gaps) == 0
        }
    
    def analyze_experiment(self, exp_path):
        """Analyze all sensors in an experiment."""
        exp_name = str(Path(exp_path).relative_to(Path(exp_path).parent.parent.parent))
        results = {}
        
        # Analyze GPS
        gps_dir = Path(exp_path) / "GPS"
        if gps_dir.exists():
            gps_files = list(gps_dir.glob("GPS_*.csv"))
            if gps_files:
                result = self.analyze_csv_timestamps(gps_files[0], 'gps')
                if 'error' not in result:
                    results['gps'] = result
        
        # Analyze IMU sensors
        imu_dir = Path(exp_path) / "IMU"
        if imu_dir.exists():
            for sensor_dir in imu_dir.iterdir():
                if sensor_dir.is_dir():
                    sensor_name = sensor_dir.name.lower().replace('sensor_', 'sensor_')
                    # Try to find accel files (most common)
                    accel_files = list(sensor_dir.glob("accel_*.csv"))
                    if accel_files:
                        result = self.analyze_csv_timestamps(accel_files[0], 'default')
                        if 'error' not in result:
                            results[sensor_name] = result
        
        return exp_name, results

def main():
    """Run analysis on all experiments."""
    analyzer = BasicTimestampAnalyzer()
    data_path = Path(__file__).parent / "02_Evaluation_Experiments"
    
    # Find all experiments
    experiments = []
    for root, dirs, files in os.walk(data_path):
        root_path = Path(root)
        if (root_path / "GPS").is_dir() and (root_path / "IMU").is_dir():
            experiments.append(root_path)
    
    print(f"Found {len(experiments)} experiments to analyze\n")
    
    # Analyze each experiment
    all_results = {}
    summary_stats = {
        'total_sensors': 0,
        'sensors_passed': 0,
        'experiments_fully_passed': 0
    }
    
    for exp_path in experiments:
        exp_name, results = analyzer.analyze_experiment(exp_path)
        all_results[exp_name] = results
        
        # Print summary for this experiment
        print(f"\nExperiment: {exp_name}")
        print("-" * 80)
        
        exp_passed = True
        for sensor_name, result in sorted(results.items()):
            status = "✓ PASS" if result['within_spec'] else "✗ FAIL"
            print(f"{sensor_name:<15} {status:<8} "
                  f"Rate: {result['actual_rate_hz']:>6.1f}Hz "
                  f"(expected {result['expected_rate_hz']:>3.0f}Hz), "
                  f"Jitter: {result['mean_jitter_ms']:>5.1f}ms "
                  f"(max {result['max_jitter_ms']:>5.1f}ms), "
                  f"Gaps: {result['num_gaps']}")
            
            summary_stats['total_sensors'] += 1
            if result['within_spec']:
                summary_stats['sensors_passed'] += 1
            else:
                exp_passed = False
        
        if exp_passed and results:
            summary_stats['experiments_fully_passed'] += 1
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total experiments analyzed: {len(experiments)}")
    print(f"Experiments fully passed: {summary_stats['experiments_fully_passed']}/{len(experiments)}")
    print(f"Total sensors analyzed: {summary_stats['total_sensors']}")
    print(f"Sensors within spec: {summary_stats['sensors_passed']}/{summary_stats['total_sensors']} "
          f"({summary_stats['sensors_passed']/summary_stats['total_sensors']*100:.1f}%)")
    
    # Save results
    output_dir = Path("timestamp_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results as JSON
    results_file = output_dir / "basic_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'analysis_time': datetime.now().isoformat(),
            'summary': summary_stats,
            'experiments': all_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create a simple CSV summary
    csv_file = output_dir / "basic_analysis_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Sensor', 'Pass/Fail', 'Rate_Hz', 'Mean_Jitter_ms', 'Max_Jitter_ms', 'Gaps'])
        
        for exp_name, results in all_results.items():
            for sensor_name, result in results.items():
                writer.writerow([
                    exp_name,
                    sensor_name,
                    'PASS' if result['within_spec'] else 'FAIL',
                    result['actual_rate_hz'],
                    result['mean_jitter_ms'],
                    result['max_jitter_ms'],
                    result['num_gaps']
                ])
    
    print(f"CSV summary saved to: {csv_file}")

if __name__ == "__main__":
    main()