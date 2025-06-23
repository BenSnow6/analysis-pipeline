"""
Rotation matrix validation for sensor orientations.
Validates that the rotation matrices correctly transform sensor data to body frame.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Add parent directory to path to import frame_definitions
sys.path.append(str(Path(__file__).parent.parent.parent))
from frame_definitions import get_R_bs_dcm, _create_R_bs_from_directions

from static_detector import StaticDetector


class RotationValidator:
    """Validates sensor rotation matrices using static gravity measurements."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the rotation validator.
        
        Args:
            config_path: Path to orientation config file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "orientation_config.yaml"
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.gravity_magnitude = self.config['physics']['gravity_m_s2']
        self.gravity_body = np.array(self.config['physics']['gravity_body_frame'])
        self.static_detector = StaticDetector(config_path)
        
    def validate_rotation_matrix(self, R: np.ndarray, tolerance: float = 1e-3) -> Dict[str, any]:
        """
        Check if a matrix is a valid rotation matrix (orthonormal).
        
        Args:
            R: 3x3 rotation matrix
            tolerance: Tolerance for orthonormality check
            
        Returns:
            Dictionary with validation results
        """
        # Check shape
        if R.shape != (3, 3):
            return {'valid': False, 'error': 'Not a 3x3 matrix'}
            
        # Check orthonormality: R @ R.T should be identity
        identity_error = np.linalg.norm(R @ R.T - np.eye(3))
        
        # Check determinant (should be +1 for proper rotation)
        det = np.linalg.det(R)
        
        return {
            'valid': identity_error < tolerance and abs(det - 1.0) < tolerance,
            'orthonormality_error': identity_error,
            'determinant': det,
            'is_proper_rotation': abs(det - 1.0) < tolerance
        }
        
    def extract_gravity_direction(self, accel_static: np.ndarray) -> np.ndarray:
        """
        Extract gravity direction from static accelerometer data.
        
        Args:
            accel_static: Static accelerometer readings (N x 3)
            
        Returns:
            Normalized gravity direction vector in sensor frame
        """
        # Average static measurements
        mean_accel = np.mean(accel_static, axis=0)
        
        # Normalize to get direction
        gravity_direction = mean_accel / np.linalg.norm(mean_accel)
        
        return gravity_direction
        
    def validate_sensor_orientation(self, 
                                  sensor_name: str,
                                  accel_data: np.ndarray,
                                  gyro_data: np.ndarray,
                                  timestamp: np.ndarray) -> Dict[str, any]:
        """
        Validate a sensor's orientation using static gravity measurements.
        
        Args:
            sensor_name: Name of the sensor
            accel_data: Accelerometer data (N x 3) in sensor frame
            gyro_data: Gyroscope data (N x 3) in sensor frame
            timestamp: Timestamps for the data
            
        Returns:
            Dictionary with validation results
        """
        results = {'sensor': sensor_name}
        
        # Get sensor configuration
        sensor_config = self.config['sensors'][sensor_name]
        tolerance_deg = (self.config['validation']['tolerances']['primary_sensors_deg'] 
                        if sensor_config['type'] == 'primary' 
                        else self.config['validation']['tolerances']['secondary_sensors_deg'])
        
        # Detect static segments
        static_segments = self.static_detector.detect_static_segments(
            timestamp, gyro_data, accel_data
        )
        
        if not static_segments:
            # If no static segments found, try to use periods of low angular velocity
            print(f"  WARNING: No static segments found for {sensor_name}")
            print(f"  Attempting to use low angular velocity periods...")
            
            # Remove NaN values
            valid_mask = ~np.any(np.isnan(gyro_data), axis=1) & ~np.any(np.isnan(accel_data), axis=1)
            valid_gyro = gyro_data[valid_mask]
            valid_accel = accel_data[valid_mask]
            valid_time = timestamp[valid_mask]
            
            if len(valid_gyro) < 100:
                results['error'] = 'Insufficient valid data for analysis'
                return results
            
            # Find periods of low angular velocity
            gyro_magnitude = np.linalg.norm(valid_gyro, axis=1)
            low_motion_threshold = np.percentile(gyro_magnitude, 10)  # Use lowest 10% of motion
            low_motion_mask = gyro_magnitude < low_motion_threshold
            
            if np.sum(low_motion_mask) < 50:
                results['error'] = 'Insufficient low-motion data for analysis'
                return results
                
            # Use low-motion data as pseudo-static
            static_data = {
                'concatenated': valid_accel[low_motion_mask],
                'segments': [(valid_time[0], valid_time[-1])]  # Fake segment for compatibility
            }
            results['warning'] = 'Using low-motion periods instead of true static segments'
            print(f"  Using {np.sum(low_motion_mask)} low-motion samples for analysis")
        else:
            # Extract static data from detected segments
            static_data = self.static_detector.get_static_data(
                timestamp, accel_data, static_segments
            )
        
        # Get gravity direction in sensor frame
        gravity_sensor = self.extract_gravity_direction(static_data['concatenated'])
        results['gravity_sensor'] = gravity_sensor
        
        # Get expected rotation matrix from frame_definitions
        R_bs_current = get_R_bs_dcm(sensor_name)
        
        # Also create rotation matrix from config
        axes = sensor_config['expected_axes']
        R_bs_config = _create_R_bs_from_directions(
            axes['x_direction'], 
            axes['y_direction'], 
            axes['z_direction']
        )
        
        # Validate both matrices
        results['R_bs_current_validation'] = self.validate_rotation_matrix(R_bs_current)
        results['R_bs_config_validation'] = self.validate_rotation_matrix(R_bs_config)
        
        # Transform gravity to body frame using both matrices
        # R_bs should transform from body to sensor, so to go from sensor to body we use R_bs.T
        gravity_body_current = R_bs_current.T @ gravity_sensor * self.gravity_magnitude
        gravity_body_config = R_bs_config.T @ gravity_sensor * self.gravity_magnitude
        
        # Expected gravity in body frame
        expected_gravity = self.gravity_body
        
        # Calculate errors
        error_current = np.arccos(np.clip(
            np.dot(gravity_body_current / np.linalg.norm(gravity_body_current),
                   expected_gravity / np.linalg.norm(expected_gravity)), -1, 1
        )) * 180 / np.pi
        
        error_config = np.arccos(np.clip(
            np.dot(gravity_body_config / np.linalg.norm(gravity_body_config),
                   expected_gravity / np.linalg.norm(expected_gravity)), -1, 1
        )) * 180 / np.pi
        
        results.update({
            'gravity_body_current': gravity_body_current,
            'gravity_body_config': gravity_body_config,
            'error_current_deg': error_current,
            'error_config_deg': error_config,
            'tolerance_deg': tolerance_deg,
            'current_matrix_valid': error_current < tolerance_deg,
            'config_matrix_valid': error_config < tolerance_deg,
            'R_bs_current': R_bs_current,
            'R_bs_config': R_bs_config,
            'num_static_segments': len(static_segments),
            'total_static_duration': sum(end - start for start, end in static_segments)
        })
        
        # Determine which matrix to recommend
        if error_config < error_current:
            results['recommended_matrix'] = 'config'
            results['recommended_R_bs'] = R_bs_config
        else:
            results['recommended_matrix'] = 'current'
            results['recommended_R_bs'] = R_bs_current
            
        return results
        
    def plot_gravity_alignment(self, validation_results: Dict[str, any], save_path: Optional[Path] = None):
        """
        Create 3D plot showing gravity vector alignment with sensor axes.
        
        Args:
            validation_results: Results from validate_sensor_orientation
            save_path: Path to save figure (optional)
        """
        fig = plt.figure(figsize=self.config['plotting']['figure_size'])
        ax = fig.add_subplot(111, projection='3d')
        
        sensor_name = validation_results['sensor']
        sensor_config = self.config['sensors'][sensor_name]
        
        # Origin
        origin = np.zeros(3)
        
        # Plot gravity vector in sensor frame (normalized)
        gravity_sensor = validation_results['gravity_sensor']
        ax.quiver(0, 0, 0, gravity_sensor[0], gravity_sensor[1], gravity_sensor[2],
                 color=self.config['plotting']['gravity_vector_color'], 
                 arrow_length_ratio=0.1, linewidth=3,
                 label='Measured Gravity')
        
        # Plot sensor axes
        scale = self.config['plotting']['vector_scale']
        colors = self.config['plotting']['sensor_axes_colors']
        
        # X axis
        ax.quiver(0, 0, 0, scale, 0, 0, color=colors['x'], 
                 arrow_length_ratio=0.1, linewidth=2,
                 label=f"X: {sensor_config['expected_axes']['x_direction']}")
        
        # Y axis  
        ax.quiver(0, 0, 0, 0, scale, 0, color=colors['y'],
                 arrow_length_ratio=0.1, linewidth=2,
                 label=f"Y: {sensor_config['expected_axes']['y_direction']}")
        
        # Z axis
        ax.quiver(0, 0, 0, 0, 0, scale, color=colors['z'],
                 arrow_length_ratio=0.1, linewidth=2,
                 label=f"Z: {sensor_config['expected_axes']['z_direction']}")
        
        # Add text annotations
        ax.text(0, 0, -0.3, f"Error: {validation_results['error_config_deg']:.1f}°", 
                fontsize=12, ha='center')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{sensor_name} - Gravity Alignment in Sensor Frame')
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        lim = 0.5
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['plotting']['dpi'])
            
        return fig, ax
        
    def plot_transformation_comparison(self, validation_results: Dict[str, any], 
                                     save_path: Optional[Path] = None):
        """
        Plot comparison of gravity vectors before and after transformation.
        
        Args:
            validation_results: Results from validate_sensor_orientation
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sensor_name = validation_results['sensor']
        
        # Left plot: Sensor frame
        gravity_sensor = validation_results['gravity_sensor'] * self.gravity_magnitude
        ax1.bar(['X', 'Y', 'Z'], gravity_sensor)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.set_title(f'{sensor_name} - Gravity in Sensor Frame')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Body frame comparison
        x = np.arange(3)
        width = 0.25
        
        expected = self.gravity_body
        current = validation_results['gravity_body_current']
        config = validation_results['gravity_body_config']
        
        ax2.bar(x - width, expected, width, label='Expected', alpha=0.8)
        ax2.bar(x, current, width, label=f'Current Matrix ({validation_results["error_current_deg"]:.1f}°)', alpha=0.8)
        ax2.bar(x + width, config, width, label=f'Config Matrix ({validation_results["error_config_deg"]:.1f}°)', alpha=0.8)
        
        ax2.set_ylabel('Acceleration (m/s²)')
        ax2.set_xlabel('Body Frame Axis')
        ax2.set_title(f'{sensor_name} - Gravity in Body Frame')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['X (Forward)', 'Y (Starboard)', 'Z (Down)'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['plotting']['dpi'])
            
        return fig, (ax1, ax2)