"""
Signal preprocessing for RPM estimation.

This module implements filtering, detrending, and signal conditioning
operations for vibration data.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def high_pass_filter(data: np.ndarray, fs: float, cutoff: float, 
                    order: int = 4) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to remove low-frequency content.
    
    Args:
        data: Input signal
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz
        order: Filter order (default: 4)
        
    Returns:
        Filtered signal
    """
    # Design Butterworth high-pass filter
    sos = signal.butter(order, cutoff, btype='highpass', fs=fs, output='sos')
    
    # Apply filter (using filtfilt for zero phase)
    filtered = signal.sosfiltfilt(sos, data)
    
    logger.debug(f"Applied HP filter: cutoff={cutoff}Hz, order={order}")
    return filtered


def compute_vibration_magnitude(accel_x: np.ndarray, accel_y: np.ndarray, 
                              accel_z: np.ndarray) -> np.ndarray:
    """
    Compute vibration magnitude from 3-axis accelerometer data.
    
    Args:
        accel_x: X-axis acceleration
        accel_y: Y-axis acceleration
        accel_z: Z-axis acceleration
        
    Returns:
        Vibration magnitude |a|
    """
    magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    return magnitude


def remove_gravity(accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray,
                  gravity: float = 9.80665) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove gravity component from accelerometer data.
    
    Args:
        accel_x: X-axis acceleration
        accel_y: Y-axis acceleration 
        accel_z: Z-axis acceleration
        gravity: Gravity constant (default: 9.80665 m/s²)
        
    Returns:
        Tuple of gravity-removed accelerations (x, y, z)
    """
    # Estimate static gravity vector (mean over signal)
    gx_mean = np.mean(accel_x)
    gy_mean = np.mean(accel_y)
    gz_mean = np.mean(accel_z)
    
    # Remove mean (gravity) component
    accel_x_nograv = accel_x - gx_mean
    accel_y_nograv = accel_y - gy_mean
    accel_z_nograv = accel_z - gz_mean
    
    logger.debug(f"Removed gravity: [{gx_mean:.3f}, {gy_mean:.3f}, {gz_mean:.3f}] m/s²")
    
    return accel_x_nograv, accel_y_nograv, accel_z_nograv


def apply_anti_alias_filter(data: np.ndarray, fs: float, cutoff: float,
                          order: int = 4) -> np.ndarray:
    """
    Apply anti-aliasing low-pass filter.
    
    Args:
        data: Input signal
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz
        order: Filter order (default: 4)
        
    Returns:
        Filtered signal
    """
    # Design Butterworth low-pass filter
    sos = signal.butter(order, cutoff, btype='lowpass', fs=fs, output='sos')
    
    # Apply filter
    filtered = signal.sosfiltfilt(sos, data)
    
    logger.debug(f"Applied anti-alias filter: cutoff={cutoff}Hz, order={order}")
    return filtered


def compute_quality_metrics(data: np.ndarray, window_size: int = None) -> dict:
    """
    Compute signal quality metrics.
    
    Args:
        data: Input signal
        window_size: Window size for rolling metrics (samples)
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        'rms': np.sqrt(np.mean(data**2)),
        'peak': np.max(np.abs(data)),
        'mean': np.mean(data),
        'std': np.std(data),
        'kurtosis': signal.kurtosis(data),
        'peak_to_rms': np.max(np.abs(data)) / np.sqrt(np.mean(data**2))
    }
    
    # Check for clipping
    max_val = np.max(np.abs(data))
    if max_val > 0:
        clipping_ratio = np.sum(np.abs(data) > 0.95 * max_val) / len(data)
        metrics['clipping_ratio'] = clipping_ratio
        metrics['is_clipped'] = clipping_ratio > 0.01  # More than 1% at max
    
    return metrics


def detrend_signal(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Remove trend from signal.
    
    Args:
        data: Input signal
        method: Detrending method ('linear', 'constant')
        
    Returns:
        Detrended signal
    """
    if method == 'linear':
        detrended = signal.detrend(data, type='linear')
    elif method == 'constant':
        detrended = signal.detrend(data, type='constant')
    else:
        raise ValueError(f"Unknown detrend method: {method}")
    
    return detrended


def preprocess_for_rpm(accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray,
                      fs: float, config: dict) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline for RPM estimation.
    
    Args:
        accel_x: X-axis acceleration
        accel_y: Y-axis acceleration
        accel_z: Z-axis acceleration
        fs: Sampling frequency
        config: Configuration dictionary
        
    Returns:
        Tuple of (processed magnitude signal, quality metrics)
    """
    # Remove gravity
    ax_nograv, ay_nograv, az_nograv = remove_gravity(accel_x, accel_y, accel_z)
    
    # Compute magnitude
    magnitude = compute_vibration_magnitude(ax_nograv, ay_nograv, az_nograv)
    
    # High-pass filter
    hp_cutoff = config.get('hp_cutoff', 5.0)
    magnitude_hp = high_pass_filter(magnitude, fs, hp_cutoff)
    
    # Detrend
    magnitude_detrend = detrend_signal(magnitude_hp, method='linear')
    
    # Compute quality metrics
    metrics = compute_quality_metrics(magnitude_detrend)
    
    logger.info(f"Preprocessing complete: RMS={metrics['rms']:.3f}, "
                f"Peak/RMS={metrics['peak_to_rms']:.2f}")
    
    return magnitude_detrend, metrics