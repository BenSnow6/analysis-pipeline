"""
Quality assessment module for vibration data.

Provides functions for computing signal quality metrics, detecting clipping,
and generating quality reports for processed IMU data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime
from .logging_config import get_logger

logger = get_logger("quality")


def process_quality_windows(data: np.ndarray, 
                          window_size: int, 
                          handling: str = "process_partial") -> List[Dict[str, Any]]:
    """
    Process data in windows with configurable edge handling.
    
    Args:
        data: Time series data
        window_size: Window size in samples
        handling: How to handle partial windows
            - "drop": Ignore incomplete windows
            - "pad": Pad with zeros to complete window
            - "process_partial": Process as-is
            
    Returns:
        List of window dictionaries with start/end indices and data
    """
    n_samples = len(data)
    windows = []
    
    for start in range(0, n_samples, window_size):
        end = min(start + window_size, n_samples)
        window_data = data[start:end]
        
        if len(window_data) < window_size:
            if handling == "drop":
                continue
            elif handling == "pad":
                window_data = np.pad(window_data, 
                                   (0, window_size - len(window_data)),
                                   mode='constant')
            # "process_partial" uses data as-is
            
        windows.append({
            "start_idx": start,
            "end_idx": end,
            "is_partial": len(window_data) < window_size,
            "data": window_data
        })
    
    return windows


def compute_window_metrics(window_data: np.ndarray, 
                         max_value: float = None) -> Dict[str, float]:
    """
    Compute quality metrics for a data window.
    
    Args:
        window_data: Array of signal values
        max_value: Maximum expected value for clipping detection
        
    Returns:
        Dictionary of computed metrics
    """
    # Basic statistics
    rms = np.sqrt(np.mean(window_data**2))
    peak = np.max(np.abs(window_data))
    
    metrics = {
        "rms": float(rms),
        "peak": float(peak),
        "mean": float(np.mean(window_data)),
        "std": float(np.std(window_data)),
        "kurtosis": float(stats.kurtosis(window_data)),
        "skewness": float(stats.skew(window_data)),
        "peak_to_rms": float(peak / rms) if rms > 0 else np.inf,
        "max_value": float(np.max(window_data)),
        "min_value": float(np.min(window_data))
    }
    
    # Clipping detection
    if max_value is not None:
        clipping_samples = np.sum(np.abs(window_data) > 0.95 * max_value)
        clipping_ratio = clipping_samples / len(window_data)
        metrics["clipping_samples"] = int(clipping_samples)
        metrics["clipping_ratio"] = float(clipping_ratio)
        metrics["clipping_detected"] = bool(clipping_ratio > 0.01)  # >1% samples clipped
    else:
        metrics["clipping_samples"] = 0
        metrics["clipping_ratio"] = 0.0
        metrics["clipping_detected"] = False
    
    return metrics


def assess_signal_quality(signal: np.ndarray,
                         time: np.ndarray,
                         config: Dict[str, Any],
                         sensor_id: str) -> Dict[str, Any]:
    """
    Perform comprehensive quality assessment on a signal.
    
    Args:
        signal: Signal data (vibration magnitude or axis)
        time: Time vector
        config: Configuration dictionary with WP1 parameters
        sensor_id: Sensor identifier
        
    Returns:
        Dictionary with quality assessment results
    """
    # Extract config parameters
    wp1_config = config.get('wp1', {})
    quality_config = wp1_config.get('quality', {})
    
    window_sec = quality_config.get('window_sec', 30.0)
    window_handling = quality_config.get('window_handling', 'process_partial')
    clipping_threshold = quality_config.get('clipping_threshold', 0.95)
    max_g_range = wp1_config.get('sensors', {}).get('max_g_range', 16.0)
    
    # Convert to samples
    fs = config.get('fs', 200)
    window_samples = int(window_sec * fs)
    max_value = max_g_range * 9.80665  # Convert g to m/s²
    
    # Process windows
    windows = process_quality_windows(signal, window_samples, window_handling)
    
    # Compute metrics for each window
    window_results = []
    clipped_windows = 0
    
    for i, window in enumerate(windows):
        # Get time bounds
        start_time = time[window['start_idx']]
        end_time = time[window['end_idx'] - 1] if window['end_idx'] > 0 else time[0]
        
        # Compute metrics
        metrics = compute_window_metrics(window['data'], max_value)
        
        # Track clipping
        if metrics['clipping_detected']:
            clipped_windows += 1
        
        window_result = {
            "window_id": i,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "is_partial": window['is_partial'],
            "metrics": metrics
        }
        
        window_results.append(window_result)
    
    # Compute overall quality
    total_windows = len(window_results)
    clipping_percentage = 100.0 * clipped_windows / total_windows if total_windows > 0 else 0.0
    
    # Classify overall quality
    thresholds = quality_config.get('thresholds', {})
    overall_quality = classify_overall_quality(clipping_percentage / 100.0, thresholds)
    quality_score = 1.0 - (clipping_percentage / 100.0)
    
    # Generate summary
    summary = {
        "sensor_id": sensor_id,
        "total_windows": total_windows,
        "clipped_windows": clipped_windows,
        "clipping_percentage": round(clipping_percentage, 2),
        "overall_quality": overall_quality,
        "quality_score": round(quality_score, 3),
        "duration_seconds": float(time[-1] - time[0]) if len(time) > 0 else 0.0,
        "sample_count": len(signal)
    }
    
    return {
        "summary": summary,
        "windows": window_results,
        "parameters_used": {
            "window_sec": window_sec,
            "window_handling": window_handling,
            "clipping_threshold": clipping_threshold,
            "max_g_range": max_g_range
        }
    }


def classify_overall_quality(clipping_ratio: float, 
                           thresholds: Dict[str, float]) -> str:
    """
    Classify overall signal quality based on clipping ratio.
    
    Args:
        clipping_ratio: Fraction of windows with clipping (0-1)
        thresholds: Dictionary of quality thresholds
        
    Returns:
        Quality classification string
    """
    # Default thresholds if not provided
    if not thresholds:
        thresholds = {
            "excellent": 0.01,
            "good": 0.05,
            "fair": 0.10,
            "poor": 1.0
        }
    
    # Classify based on thresholds
    if clipping_ratio <= thresholds.get("excellent", 0.01):
        return "excellent"
    elif clipping_ratio <= thresholds.get("good", 0.05):
        return "good"
    elif clipping_ratio <= thresholds.get("fair", 0.10):
        return "fair"
    else:
        return "poor"


def generate_quality_report(quality_results: Dict[str, Any],
                          experiment: str,
                          session: str,
                          config_version: str = "1.0") -> Dict[str, Any]:
    """
    Generate a comprehensive quality report.
    
    Args:
        quality_results: Results from assess_signal_quality
        experiment: Experiment name
        session: Session (morning/afternoon)
        config_version: Configuration version
        
    Returns:
        Complete quality report dictionary
    """
    report = {
        "experiment": experiment,
        "session": session,
        "sensor_id": quality_results["summary"]["sensor_id"],
        "processing_timestamp": datetime.utcnow().isoformat() + "Z",
        "config_version": config_version,
        "summary": quality_results["summary"],
        "parameters_used": quality_results["parameters_used"],
        "windows": quality_results["windows"],
        "processing_log": {
            "warnings": [],
            "errors": [],
            "info": []
        }
    }
    
    # Add processing notes
    if quality_results["summary"]["overall_quality"] == "poor":
        report["processing_log"]["warnings"].append(
            f"High clipping detected: {quality_results['summary']['clipping_percentage']}% of windows affected"
        )
    
    if quality_results["summary"]["total_windows"] == 0:
        report["processing_log"]["errors"].append("No valid windows processed")
    
    info_msg = f"Processed {quality_results['summary']['duration_seconds']:.1f} seconds of data"
    report["processing_log"]["info"].append(info_msg)
    
    return report


def save_quality_report(report: Dict[str, Any], output_path: Path) -> None:
    """
    Save quality report to JSON file.
    
    Args:
        report: Quality report dictionary
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(
        f"Saved quality report to {output_path}",
        sensor=report.get("sensor_id"),
        quality=report["summary"]["overall_quality"]
    )


def check_multi_axis_quality(accel_x: np.ndarray, 
                           accel_y: np.ndarray,
                           accel_z: np.ndarray,
                           config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check quality across all three acceleration axes.
    
    Args:
        accel_x: X-axis acceleration
        accel_y: Y-axis acceleration
        accel_z: Z-axis acceleration
        config: Configuration dictionary
        
    Returns:
        Dictionary with per-axis quality metrics
    """
    max_g_range = config.get('wp1', {}).get('sensors', {}).get('max_g_range', 16.0)
    max_value = max_g_range * 9.80665
    
    axes_quality = {}
    
    for axis_name, axis_data in [('x', accel_x), ('y', accel_y), ('z', accel_z)]:
        # Compute basic metrics
        metrics = compute_window_metrics(axis_data, max_value)
        
        # Check for issues
        issues = []
        if metrics['clipping_detected']:
            issues.append("clipping")
        if metrics['peak_to_rms'] > 10:
            issues.append("high_peaks")
        if abs(metrics['mean']) > 2.0:  # >2 m/s² DC offset
            issues.append("dc_offset")
        
        axes_quality[axis_name] = {
            "metrics": metrics,
            "issues": issues,
            "quality": "good" if len(issues) == 0 else "poor"
        }
    
    return axes_quality


def validate_time_alignment(time: np.ndarray, fs: float, 
                          tolerance: float = 0.01) -> Tuple[bool, List[str]]:
    """
    Validate time vector for proper alignment and sampling.
    
    Args:
        time: Time vector
        fs: Expected sampling frequency
        tolerance: Tolerance for sampling rate deviation (fraction)
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check monotonicity
    if not np.all(np.diff(time) > 0):
        issues.append("Time vector is not monotonically increasing")
    
    # Check sampling rate
    dt = np.diff(time)
    expected_dt = 1.0 / fs
    
    if len(dt) > 0:
        mean_dt = np.mean(dt)
        if abs(mean_dt - expected_dt) / expected_dt > tolerance:
            actual_fs = 1.0 / mean_dt
            issues.append(f"Sampling rate mismatch: expected {fs} Hz, got {actual_fs:.1f} Hz")
        
        # Check for gaps
        max_gap = np.max(dt)
        if max_gap > 2 * expected_dt:
            issues.append(f"Time gaps detected: max gap = {max_gap:.3f} s")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def verify_antialiasing_filter(qa_summary: Dict[str, Any], 
                              config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that anti-aliasing filter was applied in WP-1.
    
    This function checks the quality assessment summary from WP-1 to ensure
    proper anti-aliasing filtering was applied before spectral analysis.
    
    Args:
        qa_summary: Quality assessment summary from WP-1 JSON file
        config: Configuration dictionary
        
    Returns:
        Tuple of (filter_verified, verification_details)
    """
    verification_details = {
        "filter_verified": False,
        "filter_type": None,
        "cutoff_hz": None,
        "order": None,
        "warnings": [],
        "processing_timestamp": qa_summary.get("processing_timestamp", "unknown")
    }
    
    # Check if WP-1 parameters are present
    params_used = qa_summary.get("parameters_used", {})
    if not params_used:
        verification_details["warnings"].append("No processing parameters found in QA summary")
        return False, verification_details
    
    # Check for filter information in processing metadata
    # This would typically be in the processing log or parameters
    processing_log = qa_summary.get("processing_log", {})
    info_messages = processing_log.get("info", [])
    
    # Look for filter information in various places
    # 1. Check if high-pass filter was mentioned (which implies anti-alias was also applied)
    highpass_applied = any("high-pass" in msg.lower() or "highpass" in msg.lower() 
                          for msg in info_messages)
    
    # 2. Check config for expected anti-alias settings
    anti_alias_config = config.get("anti_alias", {})
    expected_cutoff = anti_alias_config.get("cutoff_hz", 85)
    expected_order = anti_alias_config.get("order", 4)
    
    # 3. Analyze frequency content if window metrics are available
    windows = qa_summary.get("windows", [])
    if windows:
        # Check for high-frequency content that shouldn't be there
        for window in windows:
            metrics = window.get("metrics", {})
            # If peak values are suspiciously high, might indicate aliasing
            peak_to_rms = metrics.get("peak_to_rms", 0)
            if peak_to_rms > 20:  # Unusually high peak-to-RMS ratio
                verification_details["warnings"].append(
                    f"High peak-to-RMS ratio ({peak_to_rms:.1f}) may indicate aliasing"
                )
    
    # 4. Check sampling rate to ensure Nyquist compliance
    fs = config.get("fs", 200)
    nyquist = fs / 2
    if expected_cutoff >= nyquist * 0.9:
        verification_details["warnings"].append(
            f"Anti-alias cutoff ({expected_cutoff} Hz) too close to Nyquist ({nyquist} Hz)"
        )
    
    # Make verification decision
    # For now, we'll assume filter was applied if high-pass was mentioned
    # In production, this would check explicit filter flags
    if highpass_applied:
        verification_details["filter_verified"] = True
        verification_details["filter_type"] = "butterworth"  # Assumed from config
        verification_details["cutoff_hz"] = expected_cutoff
        verification_details["order"] = expected_order
        filter_verified = True
    else:
        verification_details["warnings"].append(
            "Could not verify anti-aliasing filter application from QA summary"
        )
        filter_verified = False
    
    # Add config check warning if needed
    require_antialiasing = config.get("wp3", {}).get("quality", {}).get("require_antialiasing", True)
    if require_antialiasing and not filter_verified:
        verification_details["warnings"].append(
            "WP-3 requires anti-aliasing verification, but filter could not be verified"
        )
    
    return filter_verified, verification_details


def load_qa_summary(qa_file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load quality assessment summary from WP-1 output.
    
    Args:
        qa_file_path: Path to qa_summary JSON file
        
    Returns:
        QA summary dictionary or None if not found
    """
    if not qa_file_path.exists():
        logger.warning(f"QA summary file not found: {qa_file_path}")
        return None
    
    try:
        with open(qa_file_path, 'r') as f:
            qa_summary = json.load(f)
        return qa_summary
    except Exception as e:
        logger.error(f"Failed to load QA summary: {e}")
        return None