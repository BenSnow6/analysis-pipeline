"""
Orientation analysis plotting utilities.
Creates visualizations for sensor orientation validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
from src.core.paths import ORIENTATION_CONFIG_FILE


class OrientationPlotter:
    """Creates various plots for orientation analysis."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the plotter with configuration.
        
        Args:
            config_path: Path to orientation config file
        """
        if config_path is None:
            config_path = ORIENTATION_CONFIG_FILE
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_sensor_coordinate_systems(self, save_path: Optional[Path] = None):
        """
        Create 3D visualization of all sensor coordinate systems.
        
        Args:
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot craft reference frame at origin
        origin = np.zeros(3)
        scale = 2.0
        
        # Craft axes
        ax.quiver(0, 0, 0, scale, 0, 0, color='red', arrow_length_ratio=0.1, 
                 linewidth=3, label='X (Forward)')
        ax.quiver(0, 0, 0, 0, scale, 0, color='green', arrow_length_ratio=0.1,
                 linewidth=3, label='Y (Starboard)')
        ax.quiver(0, 0, 0, 0, 0, scale, color='blue', arrow_length_ratio=0.1,
                 linewidth=3, label='Z (Down)')
        
        # Plot each sensor
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, (sensor_name, sensor_config) in enumerate(self.config['sensors'].items()):
            if sensor_name == 'gps':
                continue
                
            pos = np.array(sensor_config['position_m'])
            
            # Plot sensor position
            ax.scatter(*pos, s=100, c=[colors[i]], marker='o', 
                      label=f"{sensor_name} ({sensor_config['location_description']})")
            
            # Add sensor axes (smaller scale)
            sensor_scale = 0.5
            axes_info = sensor_config['expected_axes']
            
            # Convert direction strings to vectors
            directions = {
                'forward': [1, 0, 0], 'aft': [-1, 0, 0],
                'starboard': [0, 1, 0], 'port': [0, -1, 0],
                'upward': [0, 0, -1], 'downward': [0, 0, 1]
            }
            
            # Plot sensor axes with dotted lines
            for axis, direction in [('x', axes_info['x_direction']), 
                                   ('y', axes_info['y_direction']),
                                   ('z', axes_info['z_direction'])]:
                vec = np.array(directions[direction.lower()]) * sensor_scale
                ax.quiver(pos[0], pos[1], pos[2], vec[0], vec[1], vec[2],
                         color=colors[i], arrow_length_ratio=0.2, 
                         linewidth=1, linestyle='--', alpha=0.7)
                         
        # Add craft outline (simplified)
        craft_length = self.config['craft']['dimensions']['length_m']
        craft_beam = self.config['craft']['dimensions']['beam_m']
        craft_height = self.config['craft']['dimensions']['height_m']
        
        # Draw craft bounding box
        corners = np.array([
            [-craft_length/2, -craft_beam/2, 0],
            [craft_length/2, -craft_beam/2, 0],
            [craft_length/2, craft_beam/2, 0],
            [-craft_length/2, craft_beam/2, 0],
            [-craft_length/2, -craft_beam/2, craft_height],
            [craft_length/2, -craft_beam/2, craft_height],
            [craft_length/2, craft_beam/2, craft_height],
            [-craft_length/2, craft_beam/2, craft_height]
        ])
        
        # Draw edges
        edges = [(0,1), (1,2), (2,3), (3,0),  # Bottom
                (4,5), (5,6), (6,7), (7,4),  # Top
                (0,4), (1,5), (2,6), (3,7)]  # Vertical
        
        for edge in edges:
            points = corners[list(edge)]
            ax.plot3D(*points.T, 'k-', alpha=0.3, linewidth=0.5)
            
        # Set labels and limits
        ax.set_xlabel('X (Forward) [m]')
        ax.set_ylabel('Y (Starboard) [m]')
        ax.set_zlabel('Z (Down) [m]')
        ax.set_title('Hovercraft Sensor Positions and Orientations')
        
        # Set equal aspect ratio
        max_range = max(craft_length, craft_beam, craft_height) / 2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-1, craft_height + 1])
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['plotting']['dpi'], bbox_inches='tight')
            
        return fig, ax
        
    def plot_validation_summary(self, all_experiments: Dict[str, Dict], 
                               save_path: Optional[Path] = None):
        """
        Create summary plots for multiple experiments.
        
        Args:
            all_experiments: Dictionary of experiment results
            save_path: Path to save figure
        """
        # Prepare data for plotting
        sensors = ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb']
        experiments = list(all_experiments.keys())
        
        # Create matrices for heatmaps
        rotation_errors = np.zeros((len(sensors), len(experiments)))
        static_valid = np.zeros((len(sensors), len(experiments)))
        bias_magnitudes = np.zeros((len(sensors), len(experiments)))
        dynamic_valid = np.zeros((len(sensors), len(experiments)))
        
        for j, exp_name in enumerate(experiments):
            exp_results = all_experiments[exp_name]
            if 'sensors' in exp_results:
                for i, sensor in enumerate(sensors):
                    if sensor in exp_results['sensors']:
                        sensor_results = exp_results['sensors'][sensor]
                        if 'error' not in sensor_results:
                            rotation_errors[i, j] = sensor_results.get('rotation_error_deg', np.nan)
                            static_valid[i, j] = 1 if sensor_results.get('static_valid', False) else 0
                            dynamic_valid[i, j] = 1 if sensor_results.get('dynamic_valid', False) else 0
                            
                            if 'bias_estimation' in sensor_results:
                                bias = sensor_results['bias_estimation']
                                if 'accel_bias_magnitude' in bias:
                                    bias_magnitudes[i, j] = bias['accel_bias_magnitude']
                                    
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Rotation errors heatmap
        ax = axes[0, 0]
        im = ax.imshow(rotation_errors, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.set_yticks(range(len(sensors)))
        ax.set_yticklabels(sensors)
        ax.set_title('Rotation Errors (degrees)')
        
        # Add text annotations
        for i in range(len(sensors)):
            for j in range(len(experiments)):
                text = ax.text(j, i, f'{rotation_errors[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=10)
                             
        plt.colorbar(im, ax=ax)
        
        # Validation status heatmap
        ax = axes[0, 1]
        combined_valid = static_valid * dynamic_valid
        im = ax.imshow(combined_valid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(experiments)))
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.set_yticks(range(len(sensors)))
        ax.set_yticklabels(sensors)
        ax.set_title('Overall Validation Status')
        
        # Add text annotations
        for i in range(len(sensors)):
            for j in range(len(experiments)):
                status = "✓" if combined_valid[i, j] == 1 else "✗"
                color = "green" if combined_valid[i, j] == 1 else "red"
                text = ax.text(j, i, status, ha="center", va="center", 
                             color=color, fontsize=16, weight='bold')
                             
        # Bias magnitudes bar plot
        ax = axes[1, 0]
        x = np.arange(len(sensors))
        width = 0.25
        
        for i, exp_name in enumerate(experiments):
            offset = (i - len(experiments)/2) * width
            values = bias_magnitudes[:, i]
            ax.bar(x + offset, values, width, label=exp_name, alpha=0.8)
            
        ax.set_xlabel('Sensor')
        ax.set_ylabel('Accelerometer Bias Magnitude (m/s²)')
        ax.set_title('Sensor Bias Magnitudes')
        ax.set_xticks(x)
        ax.set_xticklabels(sensors)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate overall statistics
        total_tests = np.sum(~np.isnan(rotation_errors))
        total_passed = np.sum(combined_valid)
        pass_rate = total_passed / total_tests * 100 if total_tests > 0 else 0
        
        avg_rotation_error = np.nanmean(rotation_errors)
        max_rotation_error = np.nanmax(rotation_errors)
        avg_bias = np.nanmean(bias_magnitudes)
        
        summary_text = f"""
        ORIENTATION VALIDATION SUMMARY
        
        Total Tests: {int(total_tests)}
        Tests Passed: {int(total_passed)}
        Pass Rate: {pass_rate:.1f}%
        
        Average Rotation Error: {avg_rotation_error:.2f}°
        Maximum Rotation Error: {max_rotation_error:.2f}°
        
        Average Bias Magnitude: {avg_bias:.4f} m/s²
        
        Primary Sensor Tolerance: {self.config['validation']['tolerances']['primary_sensors_deg']}°
        Secondary Sensor Tolerance: {self.config['validation']['tolerances']['secondary_sensors_deg']}°
        """
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['plotting']['dpi'])
            
        return fig, axes
        
    def plot_cross_sensor_consistency(self, experiment_results: Dict[str, any],
                                    save_path: Optional[Path] = None):
        """
        Plot cross-sensor consistency analysis.
        
        Args:
            experiment_results: Results from a single experiment
            save_path: Path to save figure
        """
        sensors = []
        gravity_vectors = []
        
        # Extract gravity vectors in body frame
        for sensor_name, sensor_results in experiment_results['sensors'].items():
            if 'error' not in sensor_results and 'rotation_validation' in sensor_results:
                sensors.append(sensor_name)
                gravity_body = sensor_results['rotation_validation'].get('gravity_body_config', 
                                                                        np.array([0, 0, 9.80665]))
                gravity_vectors.append(gravity_body / np.linalg.norm(gravity_body))
                
        n_sensors = len(sensors)
        if n_sensors < 2:
            print("Not enough sensors for cross-sensor consistency plot")
            return None
            
        # Calculate pairwise angles
        angles_matrix = np.zeros((n_sensors, n_sensors))
        
        for i in range(n_sensors):
            for j in range(n_sensors):
                if i != j:
                    angle = np.arccos(np.clip(np.dot(gravity_vectors[i], gravity_vectors[j]), -1, 1))
                    angles_matrix[i, j] = angle * 180 / np.pi
                    
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Angle difference heatmap
        im = ax1.imshow(angles_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=5)
        ax1.set_xticks(range(n_sensors))
        ax1.set_xticklabels(sensors, rotation=45, ha='right')
        ax1.set_yticks(range(n_sensors))
        ax1.set_yticklabels(sensors)
        ax1.set_title('Cross-Sensor Angle Differences (degrees)')
        
        # Add text annotations
        for i in range(n_sensors):
            for j in range(n_sensors):
                if i != j:
                    text = ax1.text(j, i, f'{angles_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=10)
                                   
        plt.colorbar(im, ax=ax1)
        
        # 3D gravity vector plot
        ax2 = fig.add_subplot(122, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_sensors))
        
        for i, (sensor, grav_vec) in enumerate(zip(sensors, gravity_vectors)):
            ax2.quiver(0, 0, 0, grav_vec[0], grav_vec[1], grav_vec[2],
                      color=colors[i], arrow_length_ratio=0.1, linewidth=2,
                      label=sensor)
                      
        # Add expected gravity direction
        expected = np.array([0, 0, 1])  # Normalized gravity in body frame
        ax2.quiver(0, 0, 0, expected[0], expected[1], expected[2],
                  color='black', arrow_length_ratio=0.1, linewidth=3,
                  linestyle='--', label='Expected')
                  
        ax2.set_xlabel('X (Forward)')
        ax2.set_ylabel('Y (Starboard)')
        ax2.set_zlabel('Z (Down)')
        ax2.set_title('Gravity Vectors in Body Frame (Normalized)')
        ax2.legend()
        
        # Set equal aspect ratio
        ax2.set_box_aspect([1,1,1])
        lim = 0.5
        ax2.set_xlim([-lim, lim])
        ax2.set_ylim([-lim, lim])
        ax2.set_zlim([-lim, lim])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['plotting']['dpi'])
            
        return fig, (ax1, ax2)