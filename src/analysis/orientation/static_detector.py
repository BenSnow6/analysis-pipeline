"""
Static segment detection for IMU data.
Identifies periods where the sensor is stationary based on gyroscope and accelerometer thresholds.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import yaml
from pathlib import Path
from src.core.paths import ORIENTATION_CONFIG_FILE


class StaticDetector:
    """Detects static (stationary) segments in IMU data."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the static detector with configuration parameters.
        
        Args:
            config_path: Path to configuration file. If None, use default config.
        """
        if config_path is None:
            config_path = ORIENTATION_CONFIG_FILE
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.gyro_threshold = config['static_detection']['gyro_threshold_rad_s']
        self.accel_std_threshold = config['static_detection']['accel_std_threshold_m_s2']
        self.min_duration = config['static_detection']['min_duration_s']
        self.window_size = config['static_detection']['window_size_s']
        
    def detect_static_segments(self, 
                             timestamp: np.ndarray,
                             gyro: np.ndarray,
                             accel: np.ndarray,
                             sample_rate: float = 200.0) -> List[Tuple[float, float]]:
        """
        Detect static segments in IMU data.
        
        Args:
            timestamp: Array of timestamps in seconds
            gyro: Gyroscope data (N x 3) in rad/s
            accel: Accelerometer data (N x 3) in m/s²
            sample_rate: Sampling rate in Hz
            
        Returns:
            List of (start_time, end_time) tuples for static segments
        """
        window_samples = int(self.window_size * sample_rate)
        
        # Calculate gyro magnitude
        gyro_mag = np.linalg.norm(gyro, axis=1)
        
        # Initialize static mask
        static_mask = np.zeros(len(timestamp), dtype=bool)
        
        # Sliding window detection
        for i in range(len(timestamp) - window_samples + 1):
            window_gyro = gyro_mag[i:i + window_samples]
            window_accel = accel[i:i + window_samples]
            
            # Check gyro threshold
            if np.max(window_gyro) < self.gyro_threshold:
                # Check accel std deviation
                accel_std = np.std(window_accel, axis=0)
                if np.max(accel_std) < self.accel_std_threshold:
                    static_mask[i:i + window_samples] = True
                    
        # Find continuous static segments
        segments = []
        in_segment = False
        start_idx = 0
        
        for i in range(len(static_mask)):
            if static_mask[i] and not in_segment:
                start_idx = i
                in_segment = True
            elif not static_mask[i] and in_segment:
                # Check duration
                duration = timestamp[i-1] - timestamp[start_idx]
                if duration >= self.min_duration:
                    segments.append((timestamp[start_idx], timestamp[i-1]))
                in_segment = False
                
        # Handle segment extending to end
        if in_segment:
            duration = timestamp[-1] - timestamp[start_idx]
            if duration >= self.min_duration:
                segments.append((timestamp[start_idx], timestamp[-1]))
                
        return segments
    
    def get_static_data(self,
                       timestamp: np.ndarray,
                       data: np.ndarray,
                       segments: List[Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """
        Extract data from static segments.
        
        Args:
            timestamp: Array of timestamps
            data: Data array (same length as timestamp)
            segments: List of (start, end) time tuples
            
        Returns:
            Dictionary with static segment data
        """
        static_data = []
        segment_indices = []
        
        for start_time, end_time in segments:
            # Ensure mask and data have same length
            if len(timestamp) != len(data):
                print(f"Warning: timestamp length ({len(timestamp)}) != data length ({len(data)})")
                # Use the minimum length
                min_len = min(len(timestamp), len(data))
                mask = (timestamp[:min_len] >= start_time) & (timestamp[:min_len] <= end_time)
                static_data.append(data[:min_len][mask])
            else:
                mask = (timestamp >= start_time) & (timestamp <= end_time)
                static_data.append(data[mask])
            segment_indices.append(np.where(mask)[0])
            
        return {
            'segments': segments,
            'data': static_data,
            'indices': segment_indices,
            'concatenated': np.concatenate(static_data) if static_data else np.array([])
        }
        
    def plot_static_segments(self, timestamp: np.ndarray, 
                           data: np.ndarray,
                           segments: List[Tuple[float, float]],
                           data_label: str = "Data",
                           ax=None):
        """
        Plot data with static segments highlighted.
        
        Args:
            timestamp: Array of timestamps
            data: Data to plot (can be multi-dimensional)
            segments: List of static segments
            data_label: Label for the data
            ax: Matplotlib axes (created if None)
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        # Plot data
        if data.ndim == 1:
            ax.plot(timestamp, data, label=data_label, alpha=0.7)
        else:
            for i in range(data.shape[1]):
                ax.plot(timestamp, data[:, i], label=f"{data_label}_{i}", alpha=0.7)
                
        # Highlight static segments
        for start, end in segments:
            ax.axvspan(start, end, alpha=0.2, color='green', label='Static' if start == segments[0][0] else '')
            
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(data_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax