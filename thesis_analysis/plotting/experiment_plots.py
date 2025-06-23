"""
Standardized plotting functions for hovercraft experiment data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
from scipy.signal import butter, filtfilt
import folium
from folium import plugins

# Set style for consistent plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ExperimentPlotter:
    """Class to handle standardized plotting for hovercraft experiments."""
    
    def __init__(self, base_path: str, experiment_name: str):
        """
        Initialize the plotter with experiment path.
        
        Args:
            base_path: Base directory containing experiments
            experiment_name: Name of the experiment folder
        """
        self.base_path = Path(base_path)
        self.experiment_name = experiment_name
        self.experiment_path = self.base_path / experiment_name
        self.output_dir = self.experiment_path / "plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load sensor orientations if available
        self.sensor_orientations = self._load_sensor_orientations()
        
    def _load_sensor_orientations(self) -> Dict:
        """Load sensor orientation configuration."""
        orient_file = self.base_path.parent / "config" / "sensor_orientations.json"
        if orient_file.exists():
            import json
            with open(orient_file, 'r') as f:
                orientations = json.load(f)
                return {item['device_name'].lower(): item for item in orientations}
        return {}
    
    def load_gps_data(self) -> pd.DataFrame:
        """Load GPS data for the experiment."""
        gps_path = self.experiment_path / "GPS" / f"GPS_{self.experiment_name}.csv"
        if not gps_path.exists():
            raise FileNotFoundError(f"GPS file not found: {gps_path}")
        
        gps_df = pd.read_csv(gps_path)
        # Ensure time_from_sync is numeric
        gps_df['time_from_sync'] = pd.to_numeric(gps_df['time_from_sync'], errors='coerce')
        gps_df = gps_df.dropna(subset=['time_from_sync'])
        return gps_df.sort_values('time_from_sync')
    
    def load_imu_data(self, sensor_name: str, data_type: str) -> pd.DataFrame:
        """Load IMU data for a specific sensor and data type."""
        imu_path = self.experiment_path / "IMU" / sensor_name / f"{data_type}_{self.experiment_name}.csv"
        if not imu_path.exists():
            raise FileNotFoundError(f"IMU file not found: {imu_path}")
        
        imu_df = pd.read_csv(imu_path)
        imu_df['time_from_sync'] = pd.to_numeric(imu_df['time_from_sync'], errors='coerce')
        imu_df = imu_df.dropna(subset=['time_from_sync'])
        return imu_df.sort_values('time_from_sync')
    
    def plot_gps_track_with_heading(self, save=True) -> plt.Figure:
        """Plot GPS track with heading vectors."""
        gps_df = self.load_gps_data()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # GPS track
        ax1.plot(gps_df['Lng'], gps_df['Lat'], 'b-', linewidth=2, label='GPS Track')
        ax1.scatter(gps_df['Lng'].iloc[0], gps_df['Lat'].iloc[0], 
                   color='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(gps_df['Lng'].iloc[-1], gps_df['Lat'].iloc[-1], 
                   color='red', s=100, marker='s', label='End', zorder=5)
        
        # Add heading vectors every n points
        n = max(1, len(gps_df) // 50)  # Show ~50 arrows
        for idx in range(0, len(gps_df), n):
            bearing = gps_df['Bearing'].iloc[idx]
            speed = gps_df['SpeedKPH'].iloc[idx]
            
            # Calculate arrow components
            bearing_rad = np.radians(bearing)
            arrow_length = speed * 0.00001  # Scale factor
            dx = arrow_length * np.sin(bearing_rad)
            dy = arrow_length * np.cos(bearing_rad)
            
            ax1.arrow(gps_df['Lng'].iloc[idx], gps_df['Lat'].iloc[idx],
                     dx, dy, head_width=0.00002, head_length=0.00001,
                     fc='red', ec='red', alpha=0.6)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'GPS Track with Heading Vectors - {self.experiment_name}')
        ax1.legend()
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        
        # Speed and bearing over time
        ax2_twin = ax2.twinx()
        ax2.plot(gps_df['time_from_sync'], gps_df['SpeedKPH'], 'b-', label='Speed (km/h)')
        ax2_twin.plot(gps_df['time_from_sync'], gps_df['Bearing'], 'r-', label='Bearing (°)', alpha=0.7)
        
        ax2.set_xlabel('Time from sync (s)')
        ax2.set_ylabel('Speed (km/h)', color='b')
        ax2_twin.set_ylabel('Bearing (°)', color='r')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        ax2.set_title('Speed and Bearing over Time')
        ax2.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'gps_track_heading.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_imu_sensor_comparison(self, data_type='accel', save=True) -> plt.Figure:
        """Plot comparison of all IMU sensors for a given data type."""
        sensors = ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb', 'Sensor_wnb']
        available_sensors = []
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axes_labels = ['X', 'Y', 'Z']
        colors = plt.cm.tab10(np.linspace(0, 1, len(sensors)))
        
        for i, sensor in enumerate(sensors):
            try:
                df = self.load_imu_data(sensor, data_type)
                available_sensors.append(sensor)
                
                for j, axis in enumerate(['x', 'y', 'z']):
                    axes[j].plot(df['time_from_sync'], df[axis], 
                               label=sensor, color=colors[i], alpha=0.8)
                    
            except FileNotFoundError:
                continue
        
        # Configure axes
        for j, ax in enumerate(axes):
            ax.set_ylabel(f'{axes_labels[j]}-axis')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time from sync (s)')
        
        # Add title
        data_type_names = {
            'accel': 'Accelerometer',
            'gyro': 'Gyroscope',
            'mag': 'Magnetometer',
            'angle': 'Angle'
        }
        unit_map = {
            'accel': 'm/s²',
            'gyro': '°/s',
            'mag': 'μT',
            'angle': '°'
        }
        
        fig.suptitle(f'{data_type_names.get(data_type, data_type)} Data Comparison - {self.experiment_name}\n'
                    f'Units: {unit_map.get(data_type, "N/A")}', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'{data_type}_comparison.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_turn_analysis(self, save=True) -> plt.Figure:
        """Analyze turning performance."""
        gps_df = self.load_gps_data()
        
        # Calculate turn rate
        bearing_diff = np.diff(gps_df['Bearing'].values)
        # Handle wrap-around
        bearing_diff = np.where(bearing_diff > 180, bearing_diff - 360, bearing_diff)
        bearing_diff = np.where(bearing_diff < -180, bearing_diff + 360, bearing_diff)
        
        time_diff = np.diff(gps_df['time_from_sync'].values)
        turn_rate = np.zeros(len(gps_df))
        turn_rate[1:] = bearing_diff / time_diff  # degrees per second
        
        # Smooth turn rate
        from scipy.signal import savgol_filter
        if len(turn_rate) > 51:
            turn_rate_smooth = savgol_filter(turn_rate, 51, 3)
        else:
            turn_rate_smooth = turn_rate
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Turn rate
        axes[0].plot(gps_df['time_from_sync'], turn_rate, 'b-', alpha=0.3, label='Raw')
        axes[0].plot(gps_df['time_from_sync'], turn_rate_smooth, 'r-', linewidth=2, label='Smoothed')
        axes[0].set_ylabel('Turn Rate (°/s)')
        axes[0].set_title(f'Turn Analysis - {self.experiment_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Speed during turn
        axes[1].plot(gps_df['time_from_sync'], gps_df['SpeedKPH'], 'g-', linewidth=2)
        axes[1].set_ylabel('Speed (km/h)')
        axes[1].grid(True, alpha=0.3)
        
        # Turn radius estimation
        speed_ms = gps_df['SpeedKPH'].values * 0.27778  # Convert to m/s
        turn_rate_rad = np.radians(turn_rate_smooth)
        # Avoid division by zero
        turn_radius = np.where(np.abs(turn_rate_rad) > 0.001, 
                              speed_ms / np.abs(turn_rate_rad), 
                              np.inf)
        turn_radius = np.clip(turn_radius, -1000, 1000)  # Limit to reasonable values
        
        axes[2].plot(gps_df['time_from_sync'], turn_radius, 'purple', linewidth=2)
        axes[2].set_ylabel('Turn Radius (m)')
        axes[2].set_xlabel('Time from sync (s)')
        axes[2].set_ylim(0, 200)  # Focus on reasonable turn radii
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'turn_analysis.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_acceleration_analysis(self, save=True) -> plt.Figure:
        """Analyze acceleration performance."""
        gps_df = self.load_gps_data()
        
        # Calculate acceleration from speed
        speed_ms = gps_df['SpeedKPH'].values * 0.27778  # Convert to m/s
        time_diff = np.diff(gps_df['time_from_sync'].values)
        acceleration = np.zeros(len(gps_df))
        acceleration[1:] = np.diff(speed_ms) / time_diff
        
        # Load IMU acceleration data for comparison
        try:
            accel_df = self.load_imu_data('Sensor_3', 'accel')
            # Merge with GPS time
            from scipy.interpolate import interp1d
            f_x = interp1d(accel_df['time_from_sync'], accel_df['x'], 
                          bounds_error=False, fill_value=np.nan)
            f_y = interp1d(accel_df['time_from_sync'], accel_df['y'], 
                          bounds_error=False, fill_value=np.nan)
            imu_x = f_x(gps_df['time_from_sync'])
            imu_y = f_y(gps_df['time_from_sync'])
            have_imu = True
        except:
            have_imu = False
        
        fig, axes = plt.subplots(3 if have_imu else 2, 1, figsize=(12, 10), sharex=True)
        
        # GPS-derived acceleration
        axes[0].plot(gps_df['time_from_sync'], acceleration, 'b-', linewidth=2)
        axes[0].set_ylabel('Acceleration (m/s²)')
        axes[0].set_title(f'Acceleration Analysis - {self.experiment_name}')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Speed profile
        axes[1].plot(gps_df['time_from_sync'], gps_df['SpeedKPH'], 'g-', linewidth=2)
        axes[1].set_ylabel('Speed (km/h)')
        axes[1].grid(True, alpha=0.3)
        
        # IMU comparison if available
        if have_imu:
            axes[2].plot(gps_df['time_from_sync'], imu_x, label='IMU X (Forward)', alpha=0.8)
            axes[2].plot(gps_df['time_from_sync'], imu_y, label='IMU Y (Port)', alpha=0.8)
            axes[2].set_ylabel('IMU Acceleration (m/s²)')
            axes[2].set_xlabel('Time from sync (s)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[1].set_xlabel('Time from sync (s)')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'acceleration_analysis.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attitude_estimation(self, save=True) -> plt.Figure:
        """Plot attitude (roll, pitch, yaw) estimation."""
        # Try to load angle data
        try:
            angle_df = self.load_imu_data('Sensor_3', 'angle')
            have_direct_angles = True
        except:
            have_direct_angles = False
        
        # Try to estimate from accelerometer
        try:
            accel_df = self.load_imu_data('Sensor_3', 'accel')
            
            # Calculate pitch and roll from accelerometer
            accel_norm = np.sqrt(accel_df['x']**2 + accel_df['y']**2 + accel_df['z']**2)
            pitch = np.degrees(np.arcsin(-accel_df['x'] / accel_norm))
            roll = np.degrees(np.arctan2(accel_df['y'], accel_df['z']))
            
            # Apply smoothing
            window = 501
            if len(pitch) > window:
                from scipy.signal import savgol_filter
                pitch_smooth = savgol_filter(pitch, window, 3)
                roll_smooth = savgol_filter(roll, window, 3)
            else:
                pitch_smooth = pitch
                roll_smooth = roll
                
            have_accel_angles = True
        except:
            have_accel_angles = False
        
        if not have_direct_angles and not have_accel_angles:
            print(f"No attitude data available for {self.experiment_name}")
            return None
        
        fig, axes = plt.subplots(3 if have_direct_angles else 2, 1, 
                                figsize=(12, 10), sharex=True)
        
        if have_direct_angles:
            # Plot direct angle measurements
            axes[0].plot(angle_df['time_from_sync'], angle_df['x'], 
                        'b-', linewidth=2, label='Roll')
            axes[0].set_ylabel('Roll (°)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(angle_df['time_from_sync'], angle_df['y'], 
                        'g-', linewidth=2, label='Pitch')
            axes[1].set_ylabel('Pitch (°)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(angle_df['time_from_sync'], angle_df['z'], 
                        'r-', linewidth=2, label='Yaw')
            axes[2].set_ylabel('Yaw (°)')
            axes[2].set_xlabel('Time from sync (s)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
        elif have_accel_angles:
            # Plot accelerometer-derived angles
            axes[0].plot(accel_df['time_from_sync'], roll, 'b-', alpha=0.3, label='Roll (raw)')
            axes[0].plot(accel_df['time_from_sync'], roll_smooth, 'b-', linewidth=2, label='Roll (smooth)')
            axes[0].set_ylabel('Roll (°)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(accel_df['time_from_sync'], pitch, 'g-', alpha=0.3, label='Pitch (raw)')
            axes[1].plot(accel_df['time_from_sync'], pitch_smooth, 'g-', linewidth=2, label='Pitch (smooth)')
            axes[1].set_ylabel('Pitch (°)')
            axes[1].set_xlabel('Time from sync (s)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(f'Attitude Estimation - {self.experiment_name}', fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'attitude_estimation.png', 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_all_plots(self):
        """Generate all standard plots for the experiment."""
        print(f"\nGenerating plots for {self.experiment_name}...")
        
        # GPS track and heading
        try:
            self.plot_gps_track_with_heading()
            print("✓ GPS track plot completed")
        except Exception as e:
            print(f"✗ GPS track plot failed: {e}")
        
        # IMU sensor comparisons
        for data_type in ['accel', 'gyro', 'mag', 'angle']:
            try:
                self.plot_imu_sensor_comparison(data_type)
                print(f"✓ {data_type} comparison completed")
            except Exception as e:
                print(f"✗ {data_type} comparison failed: {e}")
        
        # Turn analysis
        try:
            self.plot_turn_analysis()
            print("✓ Turn analysis completed")
        except Exception as e:
            print(f"✗ Turn analysis failed: {e}")
        
        # Acceleration analysis
        try:
            self.plot_acceleration_analysis()
            print("✓ Acceleration analysis completed")
        except Exception as e:
            print(f"✗ Acceleration analysis failed: {e}")
        
        # Attitude estimation
        try:
            self.plot_attitude_estimation()
            print("✓ Attitude estimation completed")
        except Exception as e:
            print(f"✗ Attitude estimation failed: {e}")
        
        print(f"Plots saved to: {self.output_dir}")
    
    def create_interactive_map(self, save=True):
        """Create an interactive HTML map of the GPS track."""
        gps_df = self.load_gps_data()
        
        # Create map centered on mean coordinates
        m = folium.Map(location=[gps_df['Lat'].mean(), gps_df['Lng'].mean()], 
                      zoom_start=16)
        
        # Add the GPS track
        coordinates = gps_df[['Lat', 'Lng']].values.tolist()
        folium.PolyLine(coordinates, weight=3, color='blue', opacity=0.8).add_to(m)
        
        # Add start and end markers
        folium.Marker(
            location=coordinates[0],
            popup='Start',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        folium.Marker(
            location=coordinates[-1],
            popup='End',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
        
        # Add bearing arrows at intervals
        n = max(1, len(gps_df) // 30)  # Show ~30 arrows
        for idx in range(0, len(gps_df), n):
            row = gps_df.iloc[idx]
            folium.Marker(
                location=[row['Lat'], row['Lng']],
                icon=folium.Icon(
                    icon='arrow-up',
                    prefix='fa',
                    angle=int(row['Bearing'])
                ),
                popup=f"Speed: {row['SpeedKPH']:.1f} km/h<br>Bearing: {row['Bearing']:.1f}°"
            ).add_to(m)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(m)
        
        if save:
            m.save(str(self.output_dir / 'interactive_map.html'))
        
        return m


def process_all_experiments(base_path: str, experiment_list: List[str] = None):
    """Process all experiments or a specified list."""
    base_path = Path(base_path)
    
    if experiment_list is None:
        # Find all experiment folders
        experiment_list = []
        for category_dir in base_path.glob("*/"):
            if category_dir.is_dir():
                for time_dir in category_dir.glob("*/"):
                    if time_dir.is_dir() and time_dir.name in ['morning', 'afternoon']:
                        for exp_dir in time_dir.glob("*/"):
                            if exp_dir.is_dir():
                                experiment_list.append(str(exp_dir.relative_to(base_path)))
    
    print(f"Found {len(experiment_list)} experiments to process")
    
    for exp_path in experiment_list:
        try:
            plotter = ExperimentPlotter(base_path, exp_path)
            plotter.generate_all_plots()
            plotter.create_interactive_map()
        except Exception as e:
            print(f"Failed to process {exp_path}: {e}")
    
    print("\nAll experiments processed!")


if __name__ == "__main__":
    # Example usage
    base_path = "/mnt/c/Users/ben/Documents/EngD/09 Data collection/01_analysis_pipeline/analysis-pipeline/02_Evaluation_Experiments"
    
    # Process a single experiment
    # plotter = ExperimentPlotter(base_path, "1a_1_Minimum_Radius_Turn/afternoon/007_Fast_stbd_turn_1")
    # plotter.generate_all_plots()
    
    # Process all experiments
    process_all_experiments(base_path)