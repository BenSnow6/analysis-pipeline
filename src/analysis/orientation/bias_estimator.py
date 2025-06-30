"""
Sensor bias estimation from static data.
Estimates accelerometer and gyroscope biases using static segments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import sys

from src.scripts.frame_definitions import get_R_bs_dcm
from src.analysis.orientation.static_detector import StaticDetector
from src.core.paths import ORIENTATION_CONFIG_FILE


class BiasEstimator:
    """Estimates sensor biases from static data segments."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the bias estimator.
        
        Args:
            config_path: Path to orientation config file
        """
        if config_path is None:
            config_path = ORIENTATION_CONFIG_FILE
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.gravity_body = np.array(self.config['physics']['gravity_body_frame'])
        self.static_duration = self.config['bias_estimation']['static_duration_s']
        self.outlier_threshold = self.config['bias_estimation']['outlier_threshold_sigma']
        self.static_detector = StaticDetector(config_path)
        
    def remove_outliers(self, data: np.ndarray, threshold_sigma: float = 3.0) -> np.ndarray:
        """
        Remove outliers using z-score method.
        
        Args:
            data: Input data array (N x 3)
            threshold_sigma: Number of standard deviations for outlier threshold
            
        Returns:
            Data with outliers removed
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Calculate z-scores
        z_scores = np.abs((data - mean) / (std + 1e-10))
        
        # Keep only data within threshold
        mask = np.all(z_scores < threshold_sigma, axis=1)
        
        return data[mask]
        
    def estimate_biases(self, 
                       sensor_name: str,
                       accel_data: np.ndarray,
                       gyro_data: np.ndarray,
                       timestamp: np.ndarray,
                       R_bs: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Estimate accelerometer and gyroscope biases.
        
        Args:
            sensor_name: Name of the sensor
            accel_data: Accelerometer data in sensor frame (m/s²)
            gyro_data: Gyroscope data in sensor frame (rad/s)
            timestamp: Timestamps
            R_bs: Rotation matrix (if None, use from config)
            
        Returns:
            Dictionary with bias estimates and statistics
        """
        results = {'sensor': sensor_name}
        
        # Get rotation matrix
        if R_bs is None:
            R_bs = get_R_bs_dcm(sensor_name)
            
        # Detect static segments
        static_segments = self.static_detector.detect_static_segments(
            timestamp, gyro_data, accel_data
        )
        
        if not static_segments:
            results['error'] = 'No static segments found'
            return results
            
        # Use first N seconds of static data
        total_static_time = 0
        indices_to_use = []
        
        for start, end in static_segments:
            mask = (timestamp >= start) & (timestamp <= end)
            indices = np.where(mask)[0]
            
            segment_duration = end - start
            if total_static_time + segment_duration <= self.static_duration:
                indices_to_use.extend(indices)
                total_static_time += segment_duration
            else:
                # Take partial segment to reach desired duration
                remaining_time = self.static_duration - total_static_time
                samples_needed = int(remaining_time * 200)  # Assuming 200 Hz
                indices_to_use.extend(indices[:samples_needed])
                break
                
        if not indices_to_use:
            results['error'] = 'Insufficient static data'
            return results
            
        # Extract static data with bounds checking
        # Ensure indices are within bounds
        max_idx = min(len(accel_data), len(gyro_data)) - 1
        valid_indices = [idx for idx in indices_to_use if idx <= max_idx]
        
        if not valid_indices:
            results['error'] = 'No valid indices after bounds checking'
            return results
            
        static_accel = accel_data[valid_indices]
        static_gyro = gyro_data[valid_indices]
        
        # Remove outliers
        static_accel_clean = self.remove_outliers(static_accel, self.outlier_threshold)
        static_gyro_clean = self.remove_outliers(static_gyro, self.outlier_threshold)
        
        results['samples_used'] = len(static_accel_clean)
        results['outliers_removed'] = len(static_accel) - len(static_accel_clean)
        results['static_duration_used'] = total_static_time
        
        # Estimate gyroscope bias (should be zero when static)
        gyro_bias_sensor = np.mean(static_gyro_clean, axis=0)
        results['gyro_bias_sensor_rad_s'] = gyro_bias_sensor
        
        # Transform gyro bias to body frame
        gyro_bias_body = R_bs @ gyro_bias_sensor
        results['gyro_bias_body_rad_s'] = gyro_bias_body
        
        # For accelerometer, transform to body frame first
        accel_body_static = np.array([R_bs @ static_accel_clean[i] for i in range(len(static_accel_clean))])
        
        # Expected acceleration in body frame when static
        expected_accel_body = self.gravity_body
        
        # Calculate bias in body frame
        mean_accel_body = np.mean(accel_body_static, axis=0)
        accel_bias_body = mean_accel_body - expected_accel_body
        results['accel_bias_body_m_s2'] = accel_bias_body
        
        # Transform bias back to sensor frame for storage
        R_sb = R_bs.T  # Body to sensor
        accel_bias_sensor = R_sb @ accel_bias_body
        results['accel_bias_sensor_m_s2'] = accel_bias_sensor
        
        # Calculate statistics
        results['accel_std_sensor'] = np.std(static_accel_clean, axis=0)
        results['gyro_std_sensor'] = np.std(static_gyro_clean, axis=0)
        results['accel_std_body'] = np.std(accel_body_static, axis=0)
        
        # Quality metrics
        results['accel_bias_magnitude'] = np.linalg.norm(accel_bias_sensor)
        results['gyro_bias_magnitude'] = np.linalg.norm(gyro_bias_sensor)
        
        # Check if biases are reasonable
        results['accel_bias_reasonable'] = results['accel_bias_magnitude'] < 0.5  # m/s²
        results['gyro_bias_reasonable'] = results['gyro_bias_magnitude'] < 0.01  # rad/s
        
        return results
        
    def plot_bias_estimation(self, 
                           sensor_name: str,
                           accel_data: np.ndarray,
                           gyro_data: np.ndarray,
                           timestamp: np.ndarray,
                           bias_results: Dict[str, any],
                           save_path: Optional[Path] = None):
        """
        Plot bias estimation results.
        
        Args:
            sensor_name: Name of the sensor
            accel_data: Original accelerometer data
            gyro_data: Original gyroscope data
            timestamp: Timestamps
            bias_results: Results from estimate_biases
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Detect static segments for visualization
        static_segments = self.static_detector.detect_static_segments(
            timestamp, gyro_data, accel_data
        )
        
        # Plot accelerometer data with static segments
        ax = axes[0, 0]
        for i, label in enumerate(['X', 'Y', 'Z']):
            ax.plot(timestamp, accel_data[:, i], label=f'Accel {label}', alpha=0.7)
            
        # Highlight static segments
        for start, end in static_segments:
            ax.axvspan(start, end, alpha=0.2, color='green')
            
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title(f'{sensor_name} - Accelerometer Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot gyroscope data with static segments
        ax = axes[0, 1]
        for i, label in enumerate(['X', 'Y', 'Z']):
            ax.plot(timestamp, gyro_data[:, i] * 180/np.pi, label=f'Gyro {label}', alpha=0.7)
            
        # Highlight static segments
        for start, end in static_segments:
            ax.axvspan(start, end, alpha=0.2, color='green')
            
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (°/s)')
        ax.set_title(f'{sensor_name} - Gyroscope Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot estimated biases
        ax = axes[1, 0]
        accel_bias = bias_results['accel_bias_sensor_m_s2']
        x = ['X', 'Y', 'Z']
        bars = ax.bar(x, accel_bias, color=['red', 'green', 'blue'], alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Bias (m/s²)')
        ax.set_title('Accelerometer Bias Estimates')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, accel_bias):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top')
        
        # Plot gyro biases
        ax = axes[1, 1]
        gyro_bias = bias_results['gyro_bias_sensor_rad_s'] * 180/np.pi  # Convert to deg/s
        bars = ax.bar(x, gyro_bias, color=['red', 'green', 'blue'], alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Bias (°/s)')
        ax.set_title('Gyroscope Bias Estimates')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, gyro_bias):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top')
        
        # Add summary text
        fig.text(0.5, 0.02, 
                f"Samples: {bias_results['samples_used']} | "
                f"Duration: {bias_results['static_duration_used']:.1f}s | "
                f"Outliers removed: {bias_results['outliers_removed']}", 
                ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['plotting']['dpi'])
            
        return fig, axes
        
    def apply_bias_correction(self, 
                            accel_data: np.ndarray,
                            gyro_data: np.ndarray,
                            accel_bias: np.ndarray,
                            gyro_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply bias correction to sensor data.
        
        Args:
            accel_data: Uncorrected accelerometer data
            gyro_data: Uncorrected gyroscope data
            accel_bias: Accelerometer bias to remove
            gyro_bias: Gyroscope bias to remove
            
        Returns:
            Tuple of (corrected_accel, corrected_gyro)
        """
        accel_corrected = accel_data - accel_bias
        gyro_corrected = gyro_data - gyro_bias
        
        return accel_corrected, gyro_corrected