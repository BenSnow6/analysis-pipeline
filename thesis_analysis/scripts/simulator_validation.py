"""
Simulator validation framework for comparing real hovercraft data with simulator outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy import signal
import json


class SimulatorValidator:
    """Class for validating simulator outputs against real experiment data."""
    
    def __init__(self, real_data_path: str, sim_data_path: str):
        """
        Initialize validator with paths to real and simulated data.
        
        Args:
            real_data_path: Path to real experiment data
            sim_data_path: Path to simulator output data
        """
        self.real_path = Path(real_data_path)
        self.sim_path = Path(sim_data_path)
        self.results = {}
        
    def load_trajectory_data(self, data_path: Path) -> pd.DataFrame:
        """Load trajectory data from GPS file."""
        gps_file = data_path / "gps_data.csv"
        if not gps_file.exists():
            # Try alternative naming
            gps_files = list(data_path.glob("*GPS*.csv"))
            if gps_files:
                gps_file = gps_files[0]
            else:
                raise FileNotFoundError(f"No GPS data found in {data_path}")
        
        return pd.read_csv(gps_file)
    
    def load_imu_data(self, data_path: Path, sensor_type: str) -> pd.DataFrame:
        """Load IMU data of specified type."""
        imu_file = data_path / f"imu_{sensor_type}.csv"
        if not imu_file.exists():
            # Try alternative naming
            imu_files = list(data_path.glob(f"*{sensor_type}*.csv"))
            if imu_files:
                imu_file = imu_files[0]
            else:
                return None
        
        return pd.read_csv(imu_file)
    
    def calculate_trajectory_error(self, real_gps: pd.DataFrame, 
                                 sim_gps: pd.DataFrame) -> Dict:
        """Calculate trajectory comparison metrics."""
        # Interpolate to common time base
        time_min = max(real_gps['time_from_sync'].min(), sim_gps['time_from_sync'].min())
        time_max = min(real_gps['time_from_sync'].max(), sim_gps['time_from_sync'].max())
        
        # Create common time array
        common_time = np.linspace(time_min, time_max, 1000)
        
        # Interpolate real data
        real_lat_interp = interp1d(real_gps['time_from_sync'], real_gps['Lat'], 
                                  bounds_error=False, fill_value='extrapolate')
        real_lng_interp = interp1d(real_gps['time_from_sync'], real_gps['Lng'], 
                                  bounds_error=False, fill_value='extrapolate')
        
        # Interpolate sim data
        sim_lat_interp = interp1d(sim_gps['time_from_sync'], sim_gps['Lat'], 
                                 bounds_error=False, fill_value='extrapolate')
        sim_lng_interp = interp1d(sim_gps['time_from_sync'], sim_gps['Lng'], 
                                 bounds_error=False, fill_value='extrapolate')
        
        # Get interpolated values
        real_lat = real_lat_interp(common_time)
        real_lng = real_lng_interp(common_time)
        sim_lat = sim_lat_interp(common_time)
        sim_lng = sim_lng_interp(common_time)
        
        # Convert to local coordinates (meters)
        # Using simple equirectangular approximation
        lat_center = (real_lat.mean() + sim_lat.mean()) / 2
        meters_per_deg_lat = 111320.0
        meters_per_deg_lng = meters_per_deg_lat * np.cos(np.radians(lat_center))
        
        real_x = (real_lng - real_lng[0]) * meters_per_deg_lng
        real_y = (real_lat - real_lat[0]) * meters_per_deg_lat
        sim_x = (sim_lng - sim_lng[0]) * meters_per_deg_lng
        sim_y = (sim_lat - sim_lat[0]) * meters_per_deg_lat
        
        # Calculate errors
        position_errors = np.sqrt((real_x - sim_x)**2 + (real_y - sim_y)**2)
        
        # Calculate path length
        real_path_length = np.sum(np.sqrt(np.diff(real_x)**2 + np.diff(real_y)**2))
        sim_path_length = np.sum(np.sqrt(np.diff(sim_x)**2 + np.diff(sim_y)**2))
        
        # Calculate cross-track error (simplified)
        cross_track_errors = []
        for i in range(len(sim_x)):
            # Find closest point on real trajectory
            distances = np.sqrt((real_x - sim_x[i])**2 + (real_y - sim_y[i])**2)
            cross_track_errors.append(np.min(distances))
        
        metrics = {
            'mean_position_error_m': np.mean(position_errors),
            'max_position_error_m': np.max(position_errors),
            'std_position_error_m': np.std(position_errors),
            'mean_cross_track_error_m': np.mean(cross_track_errors),
            'max_cross_track_error_m': np.max(cross_track_errors),
            'path_length_error_m': abs(real_path_length - sim_path_length),
            'path_length_error_percent': abs(real_path_length - sim_path_length) / real_path_length * 100,
            'final_position_error_m': position_errors[-1]
        }
        
        return metrics, {
            'time': common_time,
            'real_x': real_x, 'real_y': real_y,
            'sim_x': sim_x, 'sim_y': sim_y,
            'position_errors': position_errors,
            'cross_track_errors': cross_track_errors
        }
    
    def calculate_speed_error(self, real_gps: pd.DataFrame, 
                            sim_gps: pd.DataFrame) -> Dict:
        """Calculate speed comparison metrics."""
        # Interpolate to common time base
        time_min = max(real_gps['time_from_sync'].min(), sim_gps['time_from_sync'].min())
        time_max = min(real_gps['time_from_sync'].max(), sim_gps['time_from_sync'].max())
        common_time = np.linspace(time_min, time_max, 1000)
        
        # Interpolate speeds
        real_speed_interp = interp1d(real_gps['time_from_sync'], real_gps['SpeedKPH'], 
                                   bounds_error=False, fill_value='extrapolate')
        sim_speed_interp = interp1d(sim_gps['time_from_sync'], sim_gps['SpeedKPH'], 
                                  bounds_error=False, fill_value='extrapolate')
        
        real_speed = real_speed_interp(common_time)
        sim_speed = sim_speed_interp(common_time)
        
        speed_errors = sim_speed - real_speed
        
        metrics = {
            'mean_speed_error_kph': np.mean(speed_errors),
            'mean_abs_speed_error_kph': np.mean(np.abs(speed_errors)),
            'max_speed_error_kph': np.max(np.abs(speed_errors)),
            'std_speed_error_kph': np.std(speed_errors),
            'rmse_speed_kph': np.sqrt(np.mean(speed_errors**2)),
            'max_speed_real_kph': np.max(real_speed),
            'max_speed_sim_kph': np.max(sim_speed)
        }
        
        return metrics, {
            'time': common_time,
            'real_speed': real_speed,
            'sim_speed': sim_speed,
            'speed_errors': speed_errors
        }
    
    def calculate_heading_error(self, real_gps: pd.DataFrame, 
                              sim_gps: pd.DataFrame) -> Dict:
        """Calculate heading comparison metrics."""
        # Interpolate to common time base
        time_min = max(real_gps['time_from_sync'].min(), sim_gps['time_from_sync'].min())
        time_max = min(real_gps['time_from_sync'].max(), sim_gps['time_from_sync'].max())
        common_time = np.linspace(time_min, time_max, 1000)
        
        # Interpolate bearings
        real_bearing_interp = interp1d(real_gps['time_from_sync'], real_gps['Bearing'], 
                                     bounds_error=False, fill_value='extrapolate')
        sim_bearing_interp = interp1d(sim_gps['time_from_sync'], sim_gps['Bearing'], 
                                    bounds_error=False, fill_value='extrapolate')
        
        real_bearing = real_bearing_interp(common_time)
        sim_bearing = sim_bearing_interp(common_time)
        
        # Calculate angular difference properly
        heading_errors = sim_bearing - real_bearing
        heading_errors = np.where(heading_errors > 180, heading_errors - 360, heading_errors)
        heading_errors = np.where(heading_errors < -180, heading_errors + 360, heading_errors)
        
        # Calculate turn rates
        real_turn_rate = np.diff(real_bearing) / np.diff(common_time)
        sim_turn_rate = np.diff(sim_bearing) / np.diff(common_time)
        
        # Handle wrap-around for turn rates
        real_turn_rate = np.where(real_turn_rate > 180, real_turn_rate - 360, real_turn_rate)
        real_turn_rate = np.where(real_turn_rate < -180, real_turn_rate + 360, real_turn_rate)
        sim_turn_rate = np.where(sim_turn_rate > 180, sim_turn_rate - 360, sim_turn_rate)
        sim_turn_rate = np.where(sim_turn_rate < -180, sim_turn_rate + 360, sim_turn_rate)
        
        turn_rate_errors = sim_turn_rate - real_turn_rate
        
        metrics = {
            'mean_heading_error_deg': np.mean(np.abs(heading_errors)),
            'max_heading_error_deg': np.max(np.abs(heading_errors)),
            'std_heading_error_deg': np.std(heading_errors),
            'mean_turn_rate_error_deg_s': np.mean(np.abs(turn_rate_errors)),
            'max_turn_rate_error_deg_s': np.max(np.abs(turn_rate_errors))
        }
        
        return metrics, {
            'time': common_time,
            'real_bearing': real_bearing,
            'sim_bearing': sim_bearing,
            'heading_errors': heading_errors,
            'time_tr': common_time[1:],
            'real_turn_rate': real_turn_rate,
            'sim_turn_rate': sim_turn_rate,
            'turn_rate_errors': turn_rate_errors
        }
    
    def calculate_imu_error(self, real_imu: pd.DataFrame, 
                          sim_imu: pd.DataFrame, sensor_type: str) -> Dict:
        """Calculate IMU sensor comparison metrics."""
        if real_imu is None or sim_imu is None:
            return {}, {}
        
        # Interpolate to common time base
        time_min = max(real_imu['time_from_sync'].min(), sim_imu['time_from_sync'].min())
        time_max = min(real_imu['time_from_sync'].max(), sim_imu['time_from_sync'].max())
        common_time = np.linspace(time_min, time_max, 1000)
        
        # Interpolate each axis
        real_interp = {}
        sim_interp = {}
        errors = {}
        
        for axis in ['x', 'y', 'z']:
            real_f = interp1d(real_imu['time_from_sync'], real_imu[axis], 
                            bounds_error=False, fill_value='extrapolate')
            sim_f = interp1d(sim_imu['time_from_sync'], sim_imu[axis], 
                           bounds_error=False, fill_value='extrapolate')
            
            real_interp[axis] = real_f(common_time)
            sim_interp[axis] = sim_f(common_time)
            errors[axis] = sim_interp[axis] - real_interp[axis]
        
        # Calculate magnitude errors
        real_mag = np.sqrt(real_interp['x']**2 + real_interp['y']**2 + real_interp['z']**2)
        sim_mag = np.sqrt(sim_interp['x']**2 + sim_interp['y']**2 + sim_interp['z']**2)
        mag_errors = sim_mag - real_mag
        
        metrics = {
            f'{sensor_type}_x_rmse': np.sqrt(np.mean(errors['x']**2)),
            f'{sensor_type}_y_rmse': np.sqrt(np.mean(errors['y']**2)),
            f'{sensor_type}_z_rmse': np.sqrt(np.mean(errors['z']**2)),
            f'{sensor_type}_magnitude_rmse': np.sqrt(np.mean(mag_errors**2)),
            f'{sensor_type}_max_error': max(np.max(np.abs(errors['x'])), 
                                          np.max(np.abs(errors['y'])), 
                                          np.max(np.abs(errors['z'])))
        }
        
        return metrics, {
            'time': common_time,
            'real': real_interp,
            'sim': sim_interp,
            'errors': errors,
            'real_mag': real_mag,
            'sim_mag': sim_mag,
            'mag_errors': mag_errors
        }
    
    def plot_validation_results(self, output_dir: Path):
        """Generate comprehensive validation plots."""
        output_dir.mkdir(exist_ok=True)
        
        # Trajectory comparison
        if 'trajectory' in self.results:
            self._plot_trajectory_comparison(self.results['trajectory'], output_dir)
        
        # Speed comparison
        if 'speed' in self.results:
            self._plot_speed_comparison(self.results['speed'], output_dir)
        
        # Heading comparison
        if 'heading' in self.results:
            self._plot_heading_comparison(self.results['heading'], output_dir)
        
        # IMU comparisons
        for sensor_type in ['accel', 'gyro', 'mag']:
            if sensor_type in self.results:
                self._plot_imu_comparison(self.results[sensor_type], 
                                        sensor_type, output_dir)
    
    def _plot_trajectory_comparison(self, data: Dict, output_dir: Path):
        """Plot trajectory comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Trajectory plot
        ax1.plot(data['real_x'], data['real_y'], 'b-', linewidth=2, label='Real')
        ax1.plot(data['sim_x'], data['sim_y'], 'r--', linewidth=2, label='Simulated')
        ax1.scatter(data['real_x'][0], data['real_y'][0], color='green', 
                   s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(data['real_x'][-1], data['real_y'][-1], color='red', 
                   s=100, marker='s', label='End', zorder=5)
        ax1.set_xlabel('East (m)')
        ax1.set_ylabel('North (m)')
        ax1.set_title('Trajectory Comparison')
        ax1.legend()
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        
        # Error over time
        ax2.plot(data['time'], data['position_errors'], 'b-', label='Position Error')
        ax2.plot(data['time'], data['cross_track_errors'], 'r-', label='Cross-track Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (m)')
        ax2.set_title('Position Errors Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_speed_comparison(self, data: Dict, output_dir: Path):
        """Plot speed comparison."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Speed profiles
        ax1.plot(data['time'], data['real_speed'], 'b-', linewidth=2, label='Real')
        ax1.plot(data['time'], data['sim_speed'], 'r--', linewidth=2, label='Simulated')
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_title('Speed Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speed errors
        ax2.plot(data['time'], data['speed_errors'], 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed Error (km/h)')
        ax2.set_title('Speed Error (Sim - Real)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_heading_comparison(self, data: Dict, output_dir: Path):
        """Plot heading comparison."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Heading profiles
        ax1.plot(data['time'], data['real_bearing'], 'b-', linewidth=2, label='Real')
        ax1.plot(data['time'], data['sim_bearing'], 'r--', linewidth=2, label='Simulated')
        ax1.set_ylabel('Bearing (°)')
        ax1.set_title('Bearing Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Heading errors
        ax2.plot(data['time'], data['heading_errors'], 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Bearing Error (°)')
        ax2.set_title('Bearing Error (Sim - Real)')
        ax2.grid(True, alpha=0.3)
        
        # Turn rate comparison
        ax3.plot(data['time_tr'], data['real_turn_rate'], 'b-', label='Real', alpha=0.7)
        ax3.plot(data['time_tr'], data['sim_turn_rate'], 'r--', label='Simulated', alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Turn Rate (°/s)')
        ax3.set_title('Turn Rate Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'heading_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_imu_comparison(self, data: Dict, sensor_type: str, output_dir: Path):
        """Plot IMU sensor comparison."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        
        # Individual axes
        for i, axis in enumerate(['x', 'y', 'z']):
            axes[i].plot(data['time'], data['real'][axis], 'b-', label='Real', alpha=0.8)
            axes[i].plot(data['time'], data['sim'][axis], 'r--', label='Simulated', alpha=0.8)
            axes[i].set_ylabel(f'{axis.upper()}-axis')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Magnitude comparison
        axes[3].plot(data['time'], data['real_mag'], 'b-', label='Real', alpha=0.8)
        axes[3].plot(data['time'], data['sim_mag'], 'r--', label='Simulated', alpha=0.8)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Magnitude')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        sensor_names = {
            'accel': 'Accelerometer',
            'gyro': 'Gyroscope',
            'mag': 'Magnetometer'
        }
        fig.suptitle(f'{sensor_names.get(sensor_type, sensor_type)} Comparison', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{sensor_type}_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_validation(self) -> Dict:
        """Run complete validation analysis."""
        print(f"Validating simulator data against real experiment...")
        
        # Load GPS data
        try:
            real_gps = self.load_trajectory_data(self.real_path)
            sim_gps = self.load_trajectory_data(self.sim_path)
            
            # Trajectory validation
            traj_metrics, traj_data = self.calculate_trajectory_error(real_gps, sim_gps)
            self.results['trajectory'] = traj_data
            self.results['trajectory_metrics'] = traj_metrics
            
            # Speed validation
            speed_metrics, speed_data = self.calculate_speed_error(real_gps, sim_gps)
            self.results['speed'] = speed_data
            self.results['speed_metrics'] = speed_metrics
            
            # Heading validation
            heading_metrics, heading_data = self.calculate_heading_error(real_gps, sim_gps)
            self.results['heading'] = heading_data
            self.results['heading_metrics'] = heading_metrics
            
        except Exception as e:
            print(f"Error in GPS validation: {e}")
        
        # Load and validate IMU data
        for sensor_type in ['accel', 'gyro', 'mag']:
            try:
                real_imu = self.load_imu_data(self.real_path, sensor_type)
                sim_imu = self.load_imu_data(self.sim_path, sensor_type)
                
                if real_imu is not None and sim_imu is not None:
                    imu_metrics, imu_data = self.calculate_imu_error(real_imu, sim_imu, sensor_type)
                    self.results[sensor_type] = imu_data
                    self.results[f'{sensor_type}_metrics'] = imu_metrics
                    
            except Exception as e:
                print(f"Error in {sensor_type} validation: {e}")
        
        # Compile all metrics
        all_metrics = {}
        for key in self.results:
            if key.endswith('_metrics'):
                all_metrics.update(self.results[key])
        
        return all_metrics
    
    def generate_validation_report(self, output_path: Path) -> Dict:
        """Generate comprehensive validation report."""
        metrics = self.run_validation()
        
        # Generate plots
        plots_dir = output_path / "validation_plots"
        self.plot_validation_results(plots_dir)
        
        # Create report
        report = {
            'real_data_path': str(self.real_path),
            'sim_data_path': str(self.sim_path),
            'metrics': metrics,
            'validation_criteria': self._evaluate_validation_criteria(metrics)
        }
        
        # Save report
        with open(output_path / 'validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(report, output_path)
        
        return report
    
    def _evaluate_validation_criteria(self, metrics: Dict) -> Dict:
        """Evaluate metrics against validation criteria."""
        criteria = {
            'trajectory': {
                'mean_position_error_m': {'threshold': 5.0, 'target': 'less_than'},
                'mean_cross_track_error_m': {'threshold': 3.0, 'target': 'less_than'},
                'path_length_error_percent': {'threshold': 5.0, 'target': 'less_than'}
            },
            'speed': {
                'rmse_speed_kph': {'threshold': 5.0, 'target': 'less_than'},
                'mean_abs_speed_error_kph': {'threshold': 3.0, 'target': 'less_than'}
            },
            'heading': {
                'mean_heading_error_deg': {'threshold': 10.0, 'target': 'less_than'},
                'mean_turn_rate_error_deg_s': {'threshold': 5.0, 'target': 'less_than'}
            },
            'imu': {
                'accel_magnitude_rmse': {'threshold': 2.0, 'target': 'less_than'},
                'gyro_magnitude_rmse': {'threshold': 10.0, 'target': 'less_than'}
            }
        }
        
        results = {}
        for category, tests in criteria.items():
            results[category] = {}
            for metric, criterion in tests.items():
                if metric in metrics:
                    value = metrics[metric]
                    threshold = criterion['threshold']
                    if criterion['target'] == 'less_than':
                        passed = value < threshold
                    else:
                        passed = value > threshold
                    
                    results[category][metric] = {
                        'value': value,
                        'threshold': threshold,
                        'passed': passed
                    }
        
        return results
    
    def _generate_markdown_report(self, report: Dict, output_path: Path):
        """Generate markdown validation report."""
        md_content = """# Simulator Validation Report

## Overview

**Real Data:** {real_path}  
**Simulator Data:** {sim_path}

## Validation Results

### Summary

""".format(real_path=report['real_data_path'], sim_path=report['sim_data_path'])
        
        # Add metrics summary
        if report['metrics']:
            md_content += "| Metric | Value | Unit |\n"
            md_content += "|--------|-------|------|\n"
            
            for metric, value in sorted(report['metrics'].items()):
                unit = self._get_unit_from_metric_name(metric)
                md_content += f"| {metric.replace('_', ' ').title()} | {value:.2f} | {unit} |\n"
        
        # Add validation criteria results
        md_content += "\n### Validation Criteria\n\n"
        
        for category, results in report['validation_criteria'].items():
            md_content += f"#### {category.title()}\n\n"
            md_content += "| Metric | Value | Threshold | Status |\n"
            md_content += "|--------|-------|-----------|--------|\n"
            
            for metric, result in results.items():
                status = "✓ PASS" if result['passed'] else "✗ FAIL"
                md_content += f"| {metric.replace('_', ' ').title()} | "
                md_content += f"{result['value']:.2f} | {result['threshold']} | {status} |\n"
            
            md_content += "\n"
        
        # Save markdown
        with open(output_path / 'validation_report.md', 'w') as f:
            f.write(md_content)
    
    def _get_unit_from_metric_name(self, metric_name: str) -> str:
        """Extract unit from metric name."""
        if '_m' in metric_name and metric_name.endswith('_m'):
            return 'm'
        elif '_kph' in metric_name:
            return 'km/h'
        elif '_deg' in metric_name:
            return '°'
        elif '_deg_s' in metric_name:
            return '°/s'
        elif '_percent' in metric_name:
            return '%'
        elif '_ms2' in metric_name:
            return 'm/s²'
        else:
            return '-'


def validate_experiment(real_data_path: str, sim_data_path: str, 
                       output_path: str) -> Dict:
    """Validate a single experiment."""
    validator = SimulatorValidator(real_data_path, sim_data_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    report = validator.generate_validation_report(output_path)
    
    print(f"Validation complete. Report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage
    real_path = "/path/to/real/experiment/data"
    sim_path = "/path/to/simulator/output/data"
    output = "/path/to/validation/results"
    
    validate_experiment(real_path, sim_path, output)