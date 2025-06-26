"""
Signal quality assessment for RPM estimation.

This module provides functions for assessing signal quality,
detecting clipping, and validating time alignment.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def process_quality_windows(data: np.ndarray, window_size: int, 
                          strategy: str = "process_partial") -> List[Dict[str, Any]]:
    """
    Process data into windows for quality assessment.
    
    Args:
        data: Input data array
        window_size: Size of each window
        strategy: How to handle the last partial window
            - "pad": Pad the last window to full size
            - "process_partial": Process as-is
            - "drop": Drop partial windows
            
    Returns:
        List of window dictionaries
    """
    windows = []
    total_samples = len(data)
    
    for start in range(0, total_samples, window_size):
        end = min(start + window_size, total_samples)
        window_data = data[start:end]
        
        is_partial = (end - start) < window_size
        
        if is_partial:
            if strategy == "drop":
                continue
            elif strategy == "pad":
                # Pad with zeros to full window size
                padded = np.zeros(window_size)
                padded[:len(window_data)] = window_data
                window_data = padded
        
        windows.append({
            'data': window_data,
            'start': start,
            'end': end,
            'is_partial': is_partial
        })
    
    return windows


def compute_window_metrics(signal: np.ndarray, max_value: float = 16.0) -> Dict[str, Any]:
    """
    Compute metrics for a signal window.
    
    Args:
        signal: Signal array
        max_value: Maximum expected value for clipping detection
        
    Returns:
        Dictionary with window metrics
    """
    metrics = {
        'mean': float(np.mean(signal)),
        'std': float(np.std(signal)),
        'rms': float(np.sqrt(np.mean(signal**2))),
        'peak': float(np.max(np.abs(signal))),
        'min': float(np.min(signal)),
        'max': float(np.max(signal))
    }
    
    # Peak-to-RMS ratio
    metrics['peak_to_rms'] = metrics['peak'] / metrics['rms'] if metrics['rms'] > 0 else 0
    
    # Clipping detection
    clipping_threshold = max_value * 0.95
    clipped_samples = np.sum(np.abs(signal) >= clipping_threshold)
    
    metrics['clipping_detected'] = clipped_samples > 0
    metrics['clipping_samples'] = int(clipped_samples)
    metrics['clipping_ratio'] = float(clipped_samples / len(signal))
    
    return metrics


def compute_window_quality(signal: np.ndarray, max_g: float = 16.0,
                         clipping_threshold: float = 0.95) -> Dict[str, Any]:
    """
    Compute quality metrics for a signal window.
    
    Args:
        signal: Signal array
        max_g: Maximum g-range of the sensor
        clipping_threshold: Fraction of max range to consider clipping
        
    Returns:
        Dictionary with quality metrics
    """
    # Calculate clipping bounds
    clip_level = max_g * clipping_threshold
    
    # Count clipped samples
    clipped = np.abs(signal) >= clip_level
    clipping_ratio = np.sum(clipped) / len(signal)
    
    # Calculate other metrics
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    peak_to_rms = peak / rms if rms > 0 else 0
    
    # DC offset
    dc_offset = np.mean(signal)
    
    # Variance
    variance = np.var(signal)
    
    # Quality flag: 0=good, 1=warning, 2=bad
    if clipping_ratio > 0.01:  # >1% clipping
        quality_flag = 2
    elif clipping_ratio > 0.001:  # >0.1% clipping
        quality_flag = 1
    else:
        quality_flag = 0
    
    # Determine if window is partial (for edge handling)
    is_partial = getattr(signal, '_is_partial', False)
    
    return {
        'clipping_ratio': float(clipping_ratio),
        'rms': float(rms),
        'peak': float(peak),
        'peak_to_rms': float(peak_to_rms),
        'dc_offset': float(dc_offset),
        'variance': float(variance),
        'quality_flag': int(quality_flag),
        'is_partial': is_partial,
        'sample_count': len(signal)
    }


def assess_signal_quality(signal: np.ndarray, time: np.ndarray,
                        config: dict, sensor_id: str) -> Dict[str, Any]:
    """
    Assess signal quality over the full duration.
    
    Args:
        signal: Signal array
        time: Time array
        config: Configuration dictionary
        sensor_id: Sensor identifier
        
    Returns:
        Dictionary with quality assessment results
    """
    fs = config.get('fs', 200)
    quality_config = config.get('wp1', {}).get('quality', {})
    window_sec = quality_config.get('window_sec', 30.0)
    max_g = config.get('wp1', {}).get('sensors', {}).get('max_g_range', 16.0)
    clipping_threshold = quality_config.get('clipping_threshold', 0.95)
    
    # Calculate window size
    window_samples = int(window_sec * fs)
    
    # Process windows
    windows = []
    total_samples = len(signal)
    
    for start in range(0, total_samples, window_samples):
        end = min(start + window_samples, total_samples)
        window_signal = signal[start:end]
        window_time = time[start:end]
        
        # Mark partial windows
        if end - start < window_samples:
            window_signal = np.array(window_signal)
            window_signal._is_partial = True
        
        # Compute window quality
        window_quality = compute_window_quality(
            window_signal, max_g, clipping_threshold
        )
        
        # Add window info
        window_quality['window_id'] = len(windows)
        window_quality['start_time'] = float(window_time[0])
        window_quality['end_time'] = float(window_time[-1])
        
        windows.append(window_quality)
    
    # Calculate summary statistics
    clipped_windows = sum(1 for w in windows if w['quality_flag'] >= 2)
    warning_windows = sum(1 for w in windows if w['quality_flag'] == 1)
    good_windows = sum(1 for w in windows if w['quality_flag'] == 0)
    
    total_clipped_samples = sum(w['clipping_ratio'] * w['sample_count'] for w in windows)
    total_samples_processed = sum(w['sample_count'] for w in windows)
    
    overall_clipping_ratio = total_clipped_samples / total_samples_processed if total_samples_processed > 0 else 0
    
    # Classify overall quality
    thresholds = quality_config.get('thresholds', {
        'excellent': 0.01,
        'good': 0.05,
        'fair': 0.10,
        'poor': 1.0
    })
    
    overall_quality = classify_overall_quality(overall_clipping_ratio, thresholds)
    quality_score = 1.0 - overall_clipping_ratio
    
    summary = {
        'sensor_id': sensor_id,
        'total_windows': len(windows),
        'good_windows': good_windows,
        'warning_windows': warning_windows,
        'clipped_windows': clipped_windows,
        'clipping_percentage': overall_clipping_ratio * 100,
        'overall_quality': overall_quality,
        'quality_score': quality_score,
        'duration_seconds': float(time[-1] - time[0]),
        'sample_count': total_samples_processed
    }
    
    return {
        'summary': summary,
        'windows': windows,
        'parameters_used': {
            'window_sec': window_sec,
            'max_g': max_g,
            'clipping_threshold': clipping_threshold,
            'fs': fs
        }
    }


def classify_overall_quality(clipping_ratio: float,
                           thresholds: Dict[str, float]) -> str:
    """
    Classify overall quality based on clipping ratio.
    
    Args:
        clipping_ratio: Fraction of samples clipped
        thresholds: Dictionary of quality thresholds
        
    Returns:
        Quality classification string
    """
    if clipping_ratio <= thresholds.get('excellent', 0.01):
        return 'excellent'
    elif clipping_ratio <= thresholds.get('good', 0.05):
        return 'good'
    elif clipping_ratio <= thresholds.get('fair', 0.10):
        return 'fair'
    else:
        return 'poor'


def check_multi_axis_quality(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           config: dict) -> Dict[str, Dict[str, Any]]:
    """
    Check quality across three axes.
    
    Args:
        x, y, z: Acceleration arrays for each axis
        config: Configuration dictionary
        
    Returns:
        Dictionary with quality results for each axis
    """
    max_g = config.get('wp1', {}).get('sensors', {}).get('max_g_range', 16.0)
    
    results = {}
    
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        # Check for DC offset
        dc_offset = np.mean(axis_data)
        dc_threshold = 0.5  # g
        
        # Check for saturation
        saturation_ratio = np.sum(np.abs(axis_data) >= max_g * 0.95) / len(axis_data)
        
        # Determine quality
        issues = []
        if abs(dc_offset) > dc_threshold:
            issues.append('dc_offset')
        if saturation_ratio > 0.01:
            issues.append('saturation')
        
        # Overall quality
        if len(issues) == 0:
            quality = 'good'
        elif len(issues) == 1:
            quality = 'fair'
        else:
            quality = 'poor'
        
        results[axis_name] = {
            'quality': quality,
            'issues': issues,
            'dc_offset': float(dc_offset),
            'saturation_ratio': float(saturation_ratio)
        }
    
    return results


def validate_time_alignment(time: np.ndarray, expected_fs: float,
                          tolerance: float = 0.05) -> Tuple[bool, List[str]]:
    """
    Validate time vector for proper alignment.
    
    Args:
        time: Time array
        expected_fs: Expected sampling frequency
        tolerance: Tolerance for sampling rate (fraction)
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check monotonicity
    time_diff = np.diff(time)
    if not np.all(time_diff > 0):
        issues.append("Time vector is not monotonic")
    
    # Check sampling rate
    actual_fs = 1.0 / np.median(time_diff)
    fs_error = abs(actual_fs - expected_fs) / expected_fs
    
    if fs_error > tolerance:
        issues.append(f"Sampling rate mismatch: expected {expected_fs} Hz, got {actual_fs:.1f} Hz")
    
    # Check for gaps
    expected_dt = 1.0 / expected_fs
    max_gap = np.max(time_diff) if len(time_diff) > 0 else 0
    
    if max_gap > 2 * expected_dt:
        issues.append(f"Time gaps detected: max gap = {max_gap*1000:.1f} ms")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def generate_quality_report(quality_results: Dict[str, Any],
                          experiment: str, session: str) -> List[Dict[str, Any]]:
    """
    Generate quality report entries.
    
    Args:
        quality_results: Quality assessment results
        experiment: Experiment name
        session: Session name
        
    Returns:
        List of report entries
    """
    reports = []
    
    # Summary report
    summary = quality_results.get('summary', {})
    
    summary_report = {
        'experiment': experiment,
        'session': session,
        'sensor_id': summary.get('sensor_id', 'unknown'),
        'report_type': 'quality_summary',
        'overall_quality': summary.get('overall_quality', 'unknown'),
        'quality_score': summary.get('quality_score', 0.0),
        'clipping_percentage': summary.get('clipping_percentage', 0.0),
        'duration_seconds': summary.get('duration_seconds', 0.0),
        'sample_count': summary.get('sample_count', 0)
    }
    
    reports.append(summary_report)
    
    # Window reports for bad windows
    windows = quality_results.get('windows', [])
    for window in windows:
        if window.get('quality_flag', 0) >= 2:  # Bad windows
            window_report = {
                'experiment': experiment,
                'session': session,
                'sensor_id': summary.get('sensor_id', 'unknown'),
                'report_type': 'bad_window',
                'window_id': window.get('window_id', -1),
                'start_time': window.get('start_time', 0.0),
                'end_time': window.get('end_time', 0.0),
                'clipping_ratio': window.get('clipping_ratio', 0.0),
                'quality_flag': window.get('quality_flag', 0)
            }
            reports.append(window_report)
    
    return reports


def verify_antialiasing_filter(qa_summary: Dict[str, Any], 
                             config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify anti-aliasing filter effectiveness based on quality metrics.
    
    Args:
        qa_summary: Quality assessment summary
        config: Configuration with fs and anti_alias settings
        
    Returns:
        Tuple of (is_verified, details_dict)
    """
    details = {
        'verified': True,
        'warnings': [],
        'info': []
    }
    
    # Check peak-to-RMS ratios
    windows = qa_summary.get('windows', [])
    high_peak_windows = []
    
    for i, window in enumerate(windows):
        metrics = window.get('metrics', {})
        peak_to_rms = metrics.get('peak_to_rms', 0)
        
        # Flag windows with high peak-to-RMS (potential aliasing)
        if peak_to_rms > 20.0:  # Threshold for concern
            high_peak_windows.append((i, peak_to_rms))
    
    if high_peak_windows:
        details['verified'] = False
        details['warnings'].append(
            f"High peak-to-RMS ratios detected in {len(high_peak_windows)} windows, "
            f"indicating potential aliasing or transient events"
        )
        
        # Add specific window details
        for window_id, ratio in high_peak_windows[:3]:  # Show first 3
            details['warnings'].append(
                f"Window {window_id}: peak-to-RMS = {ratio:.1f}"
            )
    
    # Check if anti-alias filter is configured properly
    fs = config.get('fs', 200)
    anti_alias_config = config.get('anti_alias', {})
    cutoff_hz = anti_alias_config.get('cutoff_hz', fs/2.2)
    
    if cutoff_hz > fs/2:
        details['warnings'].append(
            f"Anti-alias cutoff ({cutoff_hz} Hz) exceeds Nyquist frequency ({fs/2} Hz)"
        )
        details['verified'] = False
    
    # Info about configuration
    details['info'].append(f"Sampling rate: {fs} Hz")
    details['info'].append(f"Anti-alias cutoff: {cutoff_hz} Hz")
    
    return details['verified'], details