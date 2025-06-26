"""
Dynamic maneuver validation for sensor orientations.
Uses known maneuver patterns to validate rotation matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import sys

from src.scripts.frame_definitions import get_R_bs_dcm
from src.core.paths import ORIENTATION_CONFIG_FILE


class DynamicValidator:
    """Validates sensor orientations using dynamic maneuver patterns."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the dynamic validator.
        
        Args:
            config_path: Path to orientation config file
        """
        if config_path is None:
            config_path = ORIENTATION_CONFIG_FILE
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.gravity_magnitude = self.config['physics']['gravity_m_s2']
        self.maneuver_specs = self.config['maneuver_validation']['experiments']
        
    def find_acceleration_phases(self, accel_body: np.ndarray, 
                               timestamp: np.ndarray,
                               min_accel: float = 0.5,
                               min_duration: float = 2.0) -> List[Tuple[float, float]]:
        """
        Find periods of significant forward acceleration.
        
        Args:
            accel_body: Acceleration in body frame (N x 3)
            timestamp: Timestamps
            min_accel: Minimum acceleration threshold (m/s²)
            min_duration: Minimum duration (seconds)
            
        Returns:
            List of (start_time, end_time) tuples
        """
        # Forward acceleration is in X direction
        forward_accel = accel_body[:, 0]
        
        # Remove gravity effect from Z component to get net acceleration
        net_z_accel = accel_body[:, 2] - self.gravity_magnitude
        
        # Find where forward acceleration is significant
        accel_mask = forward_accel > min_accel
        
        # Find continuous segments
        segments = []
        in_segment = False
        start_idx = 0
        
        for i in range(len(accel_mask)):
            if accel_mask[i] and not in_segment:
                start_idx = i
                in_segment = True
            elif not accel_mask[i] and in_segment:
                duration = timestamp[i-1] - timestamp[start_idx]
                if duration >= min_duration:
                    segments.append((timestamp[start_idx], timestamp[i-1]))
                in_segment = False
                
        # Handle segment extending to end
        if in_segment:
            duration = timestamp[-1] - timestamp[start_idx]
            if duration >= min_duration:
                segments.append((timestamp[start_idx], timestamp[-1]))
                
        return segments
        
    def find_turn_phases(self, gyro_body: np.ndarray,
                        timestamp: np.ndarray,
                        min_rate: float = 0.1,
                        min_duration: float = 2.0) -> List[Dict[str, any]]:
        """
        Find periods of turning maneuvers.
        
        Args:
            gyro_body: Angular velocity in body frame (N x 3)
            timestamp: Timestamps
            min_rate: Minimum turn rate threshold (rad/s)
            min_duration: Minimum duration (seconds)
            
        Returns:
            List of turn information dictionaries
        """
        # Yaw rate is around Z axis
        yaw_rate = gyro_body[:, 2]
        
        # Find significant turns
        turn_mask = np.abs(yaw_rate) > min_rate
        
        # Find continuous segments
        segments = []
        in_segment = False
        start_idx = 0
        
        for i in range(len(turn_mask)):
            if turn_mask[i] and not in_segment:
                start_idx = i
                in_segment = True
            elif not turn_mask[i] and in_segment:
                duration = timestamp[i-1] - timestamp[start_idx]
                if duration >= min_duration:
                    # Determine turn direction
                    mean_rate = np.mean(yaw_rate[start_idx:i])
                    direction = 'starboard' if mean_rate > 0 else 'port'
                    
                    segments.append({
                        'start_time': timestamp[start_idx],
                        'end_time': timestamp[i-1],
                        'direction': direction,
                        'mean_rate': mean_rate,
                        'duration': duration
                    })
                in_segment = False
                
        # Handle segment extending to end
        if in_segment:
            duration = timestamp[-1] - timestamp[start_idx]
            if duration >= min_duration:
                mean_rate = np.mean(yaw_rate[start_idx:])
                direction = 'starboard' if mean_rate > 0 else 'port'
                
                segments.append({
                    'start_time': timestamp[start_idx],
                    'end_time': timestamp[-1],
                    'direction': direction,
                    'mean_rate': mean_rate,
                    'duration': duration
                })
                
        return segments
        
    def validate_maneuver(self, experiment_name: str,
                         sensor_name: str,
                         accel_data: np.ndarray,
                         gyro_data: np.ndarray,
                         timestamp: np.ndarray,
                         R_bs: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Validate sensor orientation using a specific maneuver.
        
        Args:
            experiment_name: Name of the experiment
            sensor_name: Name of the sensor
            accel_data: Accelerometer data in sensor frame
            gyro_data: Gyroscope data in sensor frame
            timestamp: Timestamps
            R_bs: Rotation matrix to test (if None, use current)
            
        Returns:
            Validation results dictionary
        """
        results = {
            'experiment': experiment_name,
            'sensor': sensor_name
        }
        
        # Get rotation matrix
        if R_bs is None:
            R_bs = get_R_bs_dcm(sensor_name)
            
        # Transform to body frame
        accel_body = np.array([R_bs @ accel_data[i] for i in range(len(accel_data))])
        gyro_body = np.array([R_bs @ gyro_data[i] for i in range(len(gyro_data))])
        
        # Get expected pattern
        if experiment_name not in self.maneuver_specs:
            results['error'] = f'Unknown experiment: {experiment_name}'
            return results
            
        expected = self.maneuver_specs[experiment_name]
        results['expected_pattern'] = expected['expected_pattern']
        
        # Validate based on experiment type
        if '007' in experiment_name:  # Fast turn with acceleration
            # Find acceleration phases
            accel_phases = self.find_acceleration_phases(accel_body, timestamp)
            results['num_accel_phases'] = len(accel_phases)
            
            if accel_phases:
                # Check acceleration pattern during first phase
                start, end = accel_phases[0]
                mask = (timestamp >= start) & (timestamp <= end)
                
                mean_accel = np.mean(accel_body[mask], axis=0)
                results['mean_acceleration'] = mean_accel
                
                # Expected: positive X (forward) and positive Z (gravity)
                results['forward_accel_valid'] = mean_accel[0] > 0.3  # m/s²
                results['gravity_present'] = abs(mean_accel[2] - self.gravity_magnitude) < 1.0
                results['lateral_accel_small'] = abs(mean_accel[1]) < 0.5
                
                results['pattern_valid'] = (results['forward_accel_valid'] and 
                                          results['gravity_present'] and
                                          results['lateral_accel_small'])
                                          
        elif '016' in experiment_name:  # Straight cruise
            # Should have minimal lateral acceleration throughout
            mean_accel = np.mean(accel_body, axis=0)
            std_accel = np.std(accel_body, axis=0)
            
            results['mean_acceleration'] = mean_accel
            results['std_acceleration'] = std_accel
            
            # Expected: mainly gravity in Z, small X and Y
            results['gravity_dominant'] = abs(mean_accel[2] - self.gravity_magnitude) < 0.5
            results['forward_accel_small'] = abs(mean_accel[0]) < 0.3
            results['lateral_accel_small'] = abs(mean_accel[1]) < 0.3
            results['acceleration_stable'] = np.max(std_accel) < 0.5
            
            results['pattern_valid'] = (results['gravity_dominant'] and
                                      results['forward_accel_small'] and
                                      results['lateral_accel_small'] and
                                      results['acceleration_stable'])
                                      
        elif '021' in experiment_name:  # Quarter turn
            # Find turn phases
            turn_phases = self.find_turn_phases(gyro_body, timestamp)
            results['num_turn_phases'] = len(turn_phases)
            
            if turn_phases:
                # Analyze first significant turn
                turn = turn_phases[0]
                results['turn_info'] = turn
                
                # Get acceleration during turn
                mask = (timestamp >= turn['start_time']) & (timestamp <= turn['end_time'])
                mean_accel_turn = np.mean(accel_body[mask], axis=0)
                
                results['mean_accel_during_turn'] = mean_accel_turn
                
                # For port turn, expect positive Y (starboard) centripetal acceleration
                if 'port' in experiment_name:
                    results['centripetal_direction_valid'] = mean_accel_turn[1] > 0.2
                else:
                    results['centripetal_direction_valid'] = mean_accel_turn[1] < -0.2
                    
                results['pattern_valid'] = results['centripetal_direction_valid']
                
        else:
            results['error'] = 'Maneuver validation not implemented for this experiment'
            
        return results
        
    def plot_maneuver_validation(self, validation_results: Dict[str, any],
                                accel_body: np.ndarray,
                                gyro_body: np.ndarray,
                                timestamp: np.ndarray,
                                save_path: Optional[Path] = None):
        """
        Plot acceleration and rotation patterns during maneuver.
        
        Args:
            validation_results: Results from validate_maneuver
            accel_body: Acceleration in body frame
            gyro_body: Angular velocity in body frame
            timestamp: Timestamps
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        experiment = validation_results['experiment']
        sensor = validation_results['sensor']
        
        # Ensure arrays have same length
        min_len = min(len(timestamp), len(accel_body), len(gyro_body))
        if min_len == 0:
            plt.close(fig)
            return fig, axes
        timestamp = timestamp[:min_len]
        accel_body = accel_body[:min_len]
        gyro_body = gyro_body[:min_len]
        
        # Plot acceleration components
        ax = axes[0]
        ax.plot(timestamp, accel_body[:, 0], 'r-', label='X (Forward)', linewidth=1.5)
        ax.plot(timestamp, accel_body[:, 1], 'g-', label='Y (Starboard)', linewidth=1.5)
        ax.plot(timestamp, accel_body[:, 2], 'b-', label='Z (Down)', linewidth=1.5)
        ax.axhline(y=self.gravity_magnitude, color='b', linestyle='--', alpha=0.5, label='Expected Gravity')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title(f'{experiment} - {sensor} - Body Frame Accelerations')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot gyro components
        ax = axes[1]
        ax.plot(timestamp, gyro_body[:, 0] * 180/np.pi, 'r-', label='Roll rate', linewidth=1.5)
        ax.plot(timestamp, gyro_body[:, 1] * 180/np.pi, 'g-', label='Pitch rate', linewidth=1.5)
        ax.plot(timestamp, gyro_body[:, 2] * 180/np.pi, 'b-', label='Yaw rate', linewidth=1.5)
        ax.set_ylabel('Angular Velocity (°/s)')
        ax.set_title('Body Frame Rotation Rates')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot acceleration magnitude and highlight maneuver phases
        ax = axes[2]
        accel_mag = np.linalg.norm(accel_body, axis=1)
        ax.plot(timestamp, accel_mag, 'k-', label='Total Acceleration', linewidth=1.5)
        ax.axhline(y=self.gravity_magnitude, color='gray', linestyle='--', alpha=0.5, label='Gravity Magnitude')
        
        # Highlight detected phases based on experiment type
        if 'num_accel_phases' in validation_results and validation_results['num_accel_phases'] > 0:
            ax.axvspan(timestamp[0], timestamp[-1], alpha=0.1, color='red', label='Acceleration Phase')
        elif 'turn_info' in validation_results:
            turn = validation_results['turn_info']
            ax.axvspan(turn['start_time'], turn['end_time'], alpha=0.1, color='blue', 
                      label=f"{turn['direction'].capitalize()} Turn")
            
        ax.set_ylabel('Acceleration Magnitude (m/s²)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Total Acceleration and Maneuver Phases')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add validation result text
        valid_text = "PASS" if validation_results.get('pattern_valid', False) else "FAIL"
        color = 'green' if validation_results.get('pattern_valid', False) else 'red'
        fig.text(0.5, 0.02, f"Pattern Validation: {valid_text}", 
                ha='center', fontsize=14, color=color, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['plotting']['dpi'])
            
        return fig, axes