"""
Main orientation validation module.
Coordinates static validation, dynamic validation, and bias estimation.
"""

import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import sys

from src.scripts.frame_definitions import get_R_bs_dcm, _create_R_bs_from_directions
from src.analysis.orientation.static_detector import StaticDetector
from src.analysis.orientation.rotation_validator import RotationValidator
from src.analysis.orientation.dynamic_validator import DynamicValidator
from src.analysis.orientation.bias_estimator import BiasEstimator


class OrientationChecker:
    """Main class for comprehensive orientation validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the orientation checker.
        
        Args:
            config_path: Path to orientation config file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "orientation_config.yaml"
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize sub-modules
        self.static_detector = StaticDetector(config_path)
        self.rotation_validator = RotationValidator(config_path)
        self.dynamic_validator = DynamicValidator(config_path)
        self.bias_estimator = BiasEstimator(config_path)
        
    def load_aligned_data(self, experiment_name: str, 
                         aligned_data_dir: Optional[Path] = None) -> Dict[str, any]:
        """
        Load aligned sensor data from HDF5 or CSV files.
        
        Args:
            experiment_name: Name of the experiment
            aligned_data_dir: Directory containing aligned data
            
        Returns:
            Dictionary with sensor data
        """
        if aligned_data_dir is None:
            aligned_data_dir = Path(__file__).parent.parent / "alignment_analysis" / "aligned_data"
            
        data = {}
        
        # Try to find the CSV directory in main, morning, afternoon, or static subdirectories
        possible_dirs = [
            aligned_data_dir / f"{experiment_name}_csv",
            aligned_data_dir / "morning" / f"{experiment_name}_csv",
            aligned_data_dir / "afternoon" / f"{experiment_name}_csv",
            aligned_data_dir / "static" / "morning" / f"{experiment_name}_csv",
            aligned_data_dir / "static" / "afternoon" / f"{experiment_name}_csv"
        ]
        
        csv_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                csv_dir = dir_path
                break
                
        if csv_dir and csv_dir.exists():
                # Load timestamp from any sensor file (they should all have same timestamps)
                ref_sensor_file = csv_dir / "Sensor_3.csv"
                if ref_sensor_file.exists():
                    ref_df = pd.read_csv(ref_sensor_file)
                    if 't' in ref_df.columns:
                        data['timestamp'] = ref_df['t'].values
                    else:
                        # Fallback: try to reconstruct from metadata
                        metadata = pd.read_csv(csv_dir / "metadata.csv")
                        start_time = metadata['reference_start_time'].iloc[0]
                        rate = metadata['target_rate_hz'].iloc[0]
                        num_samples = metadata['aligned_samples'].iloc[0]
                        data['timestamp'] = np.arange(num_samples) / rate
                
                # Load each sensor
                for sensor_name in ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']:
                    sensor_file = csv_dir / f"{sensor_name}.csv"
                    if sensor_file.exists():
                        df = pd.read_csv(sensor_file)
                        sensor_data = {}
                        
                        # Skip empty dataframes (only header)
                        if len(df) == 0:
                            print(f"  WARNING: {sensor_name} has no data in {experiment_name}")
                            continue
                            
                        # Extract accelerometer columns (x, y, z)
                        if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                            # Convert from g's to m/s² (data is stored in g's)
                            sensor_data['accel'] = df[['x', 'y', 'z']].values * 9.80665
                            
                        # Extract gyroscope columns (gyro_x, gyro_y, gyro_z)
                        if 'gyro_x' in df.columns and 'gyro_y' in df.columns and 'gyro_z' in df.columns:
                            sensor_data['gyro'] = df[['gyro_x', 'gyro_y', 'gyro_z']].values
                        else:
                            # Try to load gyro data from original experiment data
                            gyro_data = self._load_gyro_data(experiment_name, sensor_name, len(df))
                            if gyro_data is not None:
                                sensor_data['gyro'] = gyro_data
                            
                        data[sensor_name] = sensor_data
        else:
            raise FileNotFoundError(f"No aligned data found for {experiment_name}")
                
        return data
    
    def _load_gyro_data(self, experiment_name: str, sensor_name: str, expected_length: int) -> Optional[np.ndarray]:
        """
        Load gyroscope data from original experiment files.
        
        Args:
            experiment_name: Name of the experiment
            sensor_name: Name of the sensor
            expected_length: Expected number of samples (for validation)
            
        Returns:
            Gyroscope data array or None if not found
        """
        # Base path to experiment data
        base_path = Path(__file__).parent.parent.parent.parent / "data/raw"
        
        # Search for the experiment in different locations
        possible_paths = [
            base_path / "1a_1_Minimum_Radius_Turn" / "afternoon" / experiment_name,
            base_path / "1a_1_Minimum_Radius_Turn" / "morning" / experiment_name,
            base_path / "1a_2_Rate_of_Turn_vs_Nosewheel_Steering_Angle" / "afternoon" / experiment_name,
            base_path / "1b_1_Ground_Acceleration_Time_and_Distance" / "afternoon" / experiment_name,
        ]
        
        for exp_path in possible_paths:
            if exp_path.exists():
                # Try both IMU/Sensor_X and Sensor_X paths
                gyro_paths = [
                    exp_path / "IMU" / sensor_name / f"gyro_{experiment_name}.csv",
                    exp_path / sensor_name / f"gyro_{experiment_name}.csv"
                ]
                
                for gyro_path in gyro_paths:
                    if gyro_path.exists():
                        try:
                            df = pd.read_csv(gyro_path)
                            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                                gyro_data = df[['x', 'y', 'z']].values
                                
                                # Verify length matches (allowing for small differences)
                                if abs(len(gyro_data) - expected_length) <= 1:
                                    # Trim or pad to match exactly
                                    if len(gyro_data) > expected_length:
                                        return gyro_data[:expected_length]
                                    elif len(gyro_data) < expected_length:
                                        # Pad with last value
                                        last_row = gyro_data[-1:]
                                        padding = np.repeat(last_row, expected_length - len(gyro_data), axis=0)
                                        return np.vstack([gyro_data, padding])
                                    else:
                                        return gyro_data
                                else:
                                    print(f"Warning: Gyro data length mismatch for {sensor_name}: "
                                          f"expected {expected_length}, got {len(gyro_data)}")
                        except Exception as e:
                            print(f"Error loading gyro data for {sensor_name}: {e}")
                            
        return None
        
    def validate_sensor(self, 
                       sensor_name: str,
                       sensor_data: Dict[str, np.ndarray],
                       timestamp: np.ndarray,
                       experiment_name: str) -> Dict[str, any]:
        """
        Perform complete validation for a single sensor.
        
        Args:
            sensor_name: Name of the sensor
            sensor_data: Dictionary with 'accel' and 'gyro' arrays
            timestamp: Timestamp array
            experiment_name: Name of the experiment
            
        Returns:
            Comprehensive validation results
        """
        results = {
            'sensor': sensor_name,
            'experiment': experiment_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check data availability
        if 'accel' not in sensor_data or 'gyro' not in sensor_data:
            results['error'] = 'Missing accelerometer or gyroscope data'
            return results
            
        accel = sensor_data['accel']
        gyro = sensor_data['gyro']
        
        # Ensure all arrays have the same length as timestamp
        min_len = min(len(timestamp), len(accel), len(gyro))
        if len(timestamp) != len(accel) or len(timestamp) != len(gyro):
            print(f"Warning: Array length mismatch - timestamp: {len(timestamp)}, "
                  f"accel: {len(accel)}, gyro: {len(gyro)}. Using minimum length: {min_len}")
            timestamp = timestamp[:min_len]
            accel = accel[:min_len]
            gyro = gyro[:min_len]
        
        # Step 1: Validate rotation matrix using static segments
        print(f"\nValidating {sensor_name} rotation matrix...")
        rotation_results = self.rotation_validator.validate_sensor_orientation(
            sensor_name, accel, gyro, timestamp
        )
        results['rotation_validation'] = rotation_results
        
        # Choose best rotation matrix
        best_R_bs = rotation_results['recommended_R_bs']
        best_source = rotation_results['recommended_matrix']
        results['rotation_matrix_source'] = best_source
        results['rotation_error_deg'] = rotation_results[f'error_{best_source}_deg']
        
        # Step 2: Estimate biases
        print(f"Estimating {sensor_name} biases...")
        bias_results = self.bias_estimator.estimate_biases(
            sensor_name, accel, gyro, timestamp, best_R_bs
        )
        results['bias_estimation'] = bias_results
        
        # Step 3: Apply bias correction
        if 'error' not in bias_results:
            accel_corrected, gyro_corrected = self.bias_estimator.apply_bias_correction(
                accel, gyro,
                bias_results['accel_bias_sensor_m_s2'],
                bias_results['gyro_bias_sensor_rad_s']
            )
        else:
            accel_corrected = accel
            gyro_corrected = gyro
            
        # Step 4: Dynamic validation
        print(f"Performing dynamic validation for {sensor_name}...")
        dynamic_results = self.dynamic_validator.validate_maneuver(
            experiment_name, sensor_name, 
            accel_corrected, gyro_corrected, 
            timestamp, best_R_bs
        )
        results['dynamic_validation'] = dynamic_results
        
        # Overall validation status
        tolerance = (self.config['validation']['tolerances']['primary_sensors_deg']
                    if self.config['sensors'][sensor_name]['type'] == 'primary'
                    else self.config['validation']['tolerances']['secondary_sensors_deg'])
        
        results['static_valid'] = results['rotation_error_deg'] < tolerance
        results['dynamic_valid'] = dynamic_results.get('pattern_valid', False)
        results['bias_valid'] = bias_results.get('accel_bias_reasonable', False) and bias_results.get('gyro_bias_reasonable', False)
        results['overall_valid'] = results['static_valid'] and results['bias_valid']
        
        return results
        
    def validate_experiment(self, experiment_name: str,
                          output_dir: Optional[Path] = None) -> Dict[str, any]:
        """
        Validate all sensors for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save results
            
        Returns:
            Dictionary with all validation results
        """
        print(f"\n{'='*60}")
        print(f"Validating experiment: {experiment_name}")
        print(f"{'='*60}")
        
        if output_dir is None:
            output_dir = Path(__file__).parent / "validation_results" / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load aligned data
        print("Loading aligned data...")
        try:
            data = self.load_aligned_data(experiment_name)
        except Exception as e:
            return {'error': f'Failed to load data: {str(e)}'}
            
        timestamp = data['timestamp']
        
        # Validate each sensor
        all_results = {
            'experiment': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'sensors': {}
        }
        
        for sensor_name in ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']:
            if sensor_name in data:
                sensor_results = self.validate_sensor(
                    sensor_name, data[sensor_name], timestamp, experiment_name
                )
                all_results['sensors'][sensor_name] = sensor_results
                
                # Generate plots
                if 'error' not in sensor_results:
                    # Rotation validation plot
                    fig, ax = self.rotation_validator.plot_gravity_alignment(
                        sensor_results['rotation_validation']
                    )
                    plt.savefig(output_dir / f"{sensor_name}_gravity_alignment.png")
                    plt.close(fig)
                    
                    # Transformation comparison plot
                    fig, axes = self.rotation_validator.plot_transformation_comparison(
                        sensor_results['rotation_validation']
                    )
                    plt.savefig(output_dir / f"{sensor_name}_transformation.png")
                    plt.close(fig)
                    
                    # Bias estimation plot
                    if 'error' not in sensor_results['bias_estimation']:
                        fig, axes = self.bias_estimator.plot_bias_estimation(
                            sensor_name, 
                            data[sensor_name]['accel'],
                            data[sensor_name]['gyro'],
                            timestamp,
                            sensor_results['bias_estimation']
                        )
                        plt.savefig(output_dir / f"{sensor_name}_bias_estimation.png")
                        plt.close(fig)
                        
                    # Dynamic validation plot
                    if 'error' not in sensor_results['dynamic_validation']:
                        # Transform data to body frame for plotting
                        R_bs = sensor_results['rotation_validation']['recommended_R_bs']
                        accel_body = np.array([R_bs @ data[sensor_name]['accel'][i] 
                                             for i in range(len(data[sensor_name]['accel']))])
                        gyro_body = np.array([R_bs @ data[sensor_name]['gyro'][i] 
                                            for i in range(len(data[sensor_name]['gyro']))])
                        
                        fig, axes = self.dynamic_validator.plot_maneuver_validation(
                            sensor_results['dynamic_validation'],
                            accel_body, gyro_body, timestamp
                        )
                        plt.savefig(output_dir / f"{sensor_name}_dynamic_validation.png")
                        plt.close(fig)
                        
        # Generate summary report
        self.generate_summary_report(all_results, output_dir)
        
        # Save detailed results as YAML
        with open(output_dir / "validation_results.yaml", 'w') as f:
            yaml.dump(all_results, f, default_flow_style=False)
            
        return all_results
        
    def generate_summary_report(self, results: Dict[str, any], output_dir: Path):
        """
        Generate a markdown summary report.
        
        Args:
            results: Complete validation results
            output_dir: Directory to save report
        """
        report_lines = [
            f"# Orientation Validation Report",
            f"**Experiment**: {results['experiment']}  ",
            f"**Generated**: {results['timestamp']}  ",
            f"",
            f"## Summary",
            f"",
            f"| Sensor | Rotation Error (°) | Static Valid | Bias Valid | Dynamic Valid | Overall Status |",
            f"|--------|-------------------|--------------|------------|---------------|----------------|"
        ]
        
        for sensor_name, sensor_results in results['sensors'].items():
            if 'error' in sensor_results:
                report_lines.append(
                    f"| {sensor_name} | ERROR | ❌ | ❌ | ❌ | ❌ ERROR |"
                )
            else:
                rotation_error = sensor_results.get('rotation_error_deg', -1)
                static_valid = "✅" if sensor_results.get('static_valid', False) else "❌"
                bias_valid = "✅" if sensor_results.get('bias_valid', False) else "❌"
                dynamic_valid = "✅" if sensor_results.get('dynamic_valid', False) else "❌"
                overall_valid = "✅ PASS" if sensor_results.get('overall_valid', False) else "❌ FAIL"
                
                report_lines.append(
                    f"| {sensor_name} | {rotation_error:.2f} | {static_valid} | "
                    f"{bias_valid} | {dynamic_valid} | {overall_valid} |"
                )
                
        # Add detailed results
        report_lines.extend([
            f"",
            f"## Detailed Results",
            f""
        ])
        
        for sensor_name, sensor_results in results['sensors'].items():
            if 'error' not in sensor_results:
                report_lines.extend([
                    f"### {sensor_name}",
                    f"",
                    f"**Rotation Validation**:",
                    f"- Matrix source: {sensor_results.get('rotation_matrix_source', 'N/A')}",
                    f"- Rotation error: {sensor_results.get('rotation_error_deg', -1):.2f}°",
                    f"- Static segments found: {sensor_results['rotation_validation'].get('num_static_segments', 0)}",
                    f"",
                    f"**Bias Estimation**:",
                ])
                
                if 'bias_estimation' in sensor_results and 'error' not in sensor_results['bias_estimation']:
                    bias = sensor_results['bias_estimation']
                    accel_bias = bias['accel_bias_sensor_m_s2']
                    gyro_bias = bias['gyro_bias_sensor_rad_s']
                    
                    report_lines.extend([
                        f"- Accelerometer bias: [{accel_bias[0]:.4f}, {accel_bias[1]:.4f}, {accel_bias[2]:.4f}] m/s²",
                        f"- Gyroscope bias: [{gyro_bias[0]:.4f}, {gyro_bias[1]:.4f}, {gyro_bias[2]:.4f}] rad/s",
                        f"- Samples used: {bias.get('samples_used', 0)}",
                        f"- Static duration: {bias.get('static_duration_used', 0):.1f} s",
                    ])
                    
                report_lines.extend([
                    f"",
                    f"**Dynamic Validation**:",
                    f"- Pattern valid: {sensor_results['dynamic_validation'].get('pattern_valid', False)}",
                    f"- Expected pattern: {sensor_results['dynamic_validation'].get('expected_pattern', 'N/A')}",
                    f""
                ])
                
        # Save report
        with open(output_dir / "VALIDATION_REPORT.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
            
        print(f"\nValidation report saved to: {output_dir / 'VALIDATION_REPORT.md'}")