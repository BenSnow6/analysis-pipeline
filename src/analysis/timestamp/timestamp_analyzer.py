"""
Core timestamp analysis functionality.

This module provides functions to analyze timestamp consistency,
calculate jitter, detect gaps, and validate against specifications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings


@dataclass
class TimestampAnalysisResult:
    """Results from timestamp analysis."""
    sensor_name: str
    num_samples: int
    duration_seconds: float
    
    # Timing statistics
    expected_rate_hz: float
    actual_rate_hz: float
    rate_deviation_percent: float
    
    # Interval statistics (in milliseconds)
    expected_interval_ms: float
    mean_interval_ms: float
    std_interval_ms: float
    min_interval_ms: float
    max_interval_ms: float
    
    # Jitter analysis
    mean_jitter_ms: float
    std_jitter_ms: float
    max_jitter_ms: float
    jitter_threshold_ms: float
    jitter_violations: int
    jitter_pass: bool
    
    # Gap detection
    gaps: List[Dict[str, Any]]  # List of gap info dicts
    num_gaps: int
    gap_threshold_ms: float
    
    # Overall assessment
    within_spec: bool
    issues: List[str]
    
    # Raw data for plotting
    timestamps: np.ndarray
    intervals: np.ndarray
    jitter: np.ndarray


def calculate_intervals(timestamps: np.ndarray) -> np.ndarray:
    """
    Calculate time intervals between consecutive timestamps.
    
    Args:
        timestamps: Array of timestamps in seconds
        
    Returns:
        Array of intervals in milliseconds
    """
    if len(timestamps) < 2:
        return np.array([])
    
    # Calculate differences and convert to milliseconds
    intervals = np.diff(timestamps) * 1000.0
    return intervals


def analyze_sampling_rate(timestamps: np.ndarray, 
                         expected_rate_hz: float) -> Tuple[float, float, float]:
    """
    Analyze actual sampling rate from timestamps.
    
    Args:
        timestamps: Array of timestamps in seconds
        expected_rate_hz: Expected sampling rate in Hz
        
    Returns:
        Tuple of (actual_rate_hz, expected_interval_ms, rate_deviation_percent)
    """
    if len(timestamps) < 2:
        return 0.0, 0.0, 100.0
    
    # Calculate actual rate from total duration
    duration = timestamps[-1] - timestamps[0]
    num_intervals = len(timestamps) - 1
    
    if duration > 0:
        actual_rate_hz = num_intervals / duration
    else:
        actual_rate_hz = 0.0
    
    expected_interval_ms = 1000.0 / expected_rate_hz if expected_rate_hz > 0 else 0.0
    
    # Calculate deviation
    if expected_rate_hz > 0:
        rate_deviation_percent = abs(actual_rate_hz - expected_rate_hz) / expected_rate_hz * 100
    else:
        rate_deviation_percent = 100.0
    
    return actual_rate_hz, expected_interval_ms, rate_deviation_percent


def calculate_jitter(intervals: np.ndarray, expected_interval_ms: float) -> np.ndarray:
    """
    Calculate jitter (deviation from expected interval).
    
    Args:
        intervals: Array of time intervals in milliseconds
        expected_interval_ms: Expected interval in milliseconds
        
    Returns:
        Array of jitter values in milliseconds
    """
    if len(intervals) == 0:
        return np.array([])
    
    # Jitter is the absolute deviation from expected interval
    jitter = np.abs(intervals - expected_interval_ms)
    return jitter


def detect_gaps(timestamps: np.ndarray, intervals: np.ndarray, 
                gap_threshold_ms: float) -> List[Dict[str, Any]]:
    """
    Detect gaps in the timestamp sequence.
    
    Args:
        timestamps: Array of timestamps in seconds
        intervals: Array of intervals in milliseconds
        gap_threshold_ms: Threshold for gap detection in milliseconds
        
    Returns:
        List of gap information dictionaries
    """
    gaps = []
    
    if len(intervals) == 0:
        return gaps
    
    # Find indices where interval exceeds threshold
    gap_indices = np.where(intervals > gap_threshold_ms)[0]
    
    for idx in gap_indices:
        gap_info = {
            'index': int(idx),
            'start_time': float(timestamps[idx]),
            'end_time': float(timestamps[idx + 1]),
            'duration_ms': float(intervals[idx]),
            'samples_before': int(idx + 1),
            'samples_after': int(len(timestamps) - idx - 1)
        }
        gaps.append(gap_info)
    
    return gaps


def validate_against_spec(analysis_result: TimestampAnalysisResult,
                         rate_tolerance_percent: float = 10.0) -> Tuple[bool, List[str]]:
    """
    Validate analysis results against specifications.
    
    Args:
        analysis_result: Analysis results to validate
        rate_tolerance_percent: Acceptable deviation in sampling rate
        
    Returns:
        Tuple of (within_spec, list_of_issues)
    """
    issues = []
    within_spec = True
    
    # Check sampling rate
    if analysis_result.rate_deviation_percent > rate_tolerance_percent:
        within_spec = False
        issues.append(f"Sampling rate deviation {analysis_result.rate_deviation_percent:.1f}% "
                     f"exceeds tolerance of {rate_tolerance_percent}%")
    
    # Check jitter
    if not analysis_result.jitter_pass:
        within_spec = False
        issues.append(f"Jitter violations: {analysis_result.jitter_violations} samples "
                     f"exceed {analysis_result.jitter_threshold_ms}ms threshold")
    
    # Check for gaps
    if analysis_result.num_gaps > 0:
        # For low-rate sensors like GPS, some gaps might be acceptable
        if analysis_result.expected_rate_hz < 10:  # Low rate sensor
            if analysis_result.num_gaps > 5:  # Arbitrary threshold
                within_spec = False
                issues.append(f"Excessive gaps: {analysis_result.num_gaps} gaps detected")
        else:  # High rate sensor
            within_spec = False
            issues.append(f"Gaps detected: {analysis_result.num_gaps} gaps > "
                         f"{analysis_result.gap_threshold_ms}ms")
    
    # Check minimum samples
    min_duration_s = 10.0  # Minimum 10 seconds of data
    if analysis_result.duration_seconds < min_duration_s:
        within_spec = False
        issues.append(f"Insufficient data: {analysis_result.duration_seconds:.1f}s "
                     f"< {min_duration_s}s minimum")
    
    return within_spec, issues


def analyze_timestamps(timestamps: np.ndarray, sensor_name: str,
                      sensor_config: Dict[str, Any],
                      analysis_config: Optional[Dict[str, Any]] = None) -> TimestampAnalysisResult:
    """
    Perform comprehensive timestamp analysis for a sensor.
    
    Args:
        timestamps: Array of timestamps in seconds
        sensor_name: Name of the sensor
        sensor_config: Sensor configuration from specs
        analysis_config: Analysis configuration settings
        
    Returns:
        TimestampAnalysisResult with complete analysis
    """
    if analysis_config is None:
        analysis_config = {
            'min_samples': 100,
            'rate_tolerance_percent': 10.0
        }
    
    # Extract configuration
    expected_rate_hz = sensor_config.get('expected_rate_hz', 100)
    jitter_threshold_ms = sensor_config.get('jitter_threshold_ms', 20)
    gap_threshold_factor = sensor_config.get('gap_threshold_factor', 10.0)
    
    # Calculate basic statistics
    num_samples = len(timestamps)
    
    if num_samples < 2:
        # Not enough data for analysis
        return TimestampAnalysisResult(
            sensor_name=sensor_name,
            num_samples=num_samples,
            duration_seconds=0.0,
            expected_rate_hz=expected_rate_hz,
            actual_rate_hz=0.0,
            rate_deviation_percent=100.0,
            expected_interval_ms=1000.0 / expected_rate_hz,
            mean_interval_ms=0.0,
            std_interval_ms=0.0,
            min_interval_ms=0.0,
            max_interval_ms=0.0,
            mean_jitter_ms=0.0,
            std_jitter_ms=0.0,
            max_jitter_ms=0.0,
            jitter_threshold_ms=jitter_threshold_ms,
            jitter_violations=0,
            jitter_pass=False,
            gaps=[],
            num_gaps=0,
            gap_threshold_ms=0.0,
            within_spec=False,
            issues=["Insufficient data for analysis"],
            timestamps=timestamps,
            intervals=np.array([]),
            jitter=np.array([])
        )
    
    # Calculate intervals
    intervals = calculate_intervals(timestamps)
    duration_seconds = timestamps[-1] - timestamps[0]
    
    # Analyze sampling rate
    actual_rate_hz, expected_interval_ms, rate_deviation_percent = \
        analyze_sampling_rate(timestamps, expected_rate_hz)
    
    # Calculate jitter
    jitter = calculate_jitter(intervals, expected_interval_ms)
    
    # Jitter statistics
    mean_jitter_ms = np.mean(jitter) if len(jitter) > 0 else 0.0
    std_jitter_ms = np.std(jitter) if len(jitter) > 0 else 0.0
    max_jitter_ms = np.max(jitter) if len(jitter) > 0 else 0.0
    
    # Count jitter violations
    jitter_violations = np.sum(jitter > jitter_threshold_ms)
    jitter_pass = jitter_violations == 0
    
    # Detect gaps
    gap_threshold_ms = expected_interval_ms * gap_threshold_factor
    gaps = detect_gaps(timestamps, intervals, gap_threshold_ms)
    
    # Create result object
    result = TimestampAnalysisResult(
        sensor_name=sensor_name,
        num_samples=num_samples,
        duration_seconds=duration_seconds,
        expected_rate_hz=expected_rate_hz,
        actual_rate_hz=actual_rate_hz,
        rate_deviation_percent=rate_deviation_percent,
        expected_interval_ms=expected_interval_ms,
        mean_interval_ms=np.mean(intervals) if len(intervals) > 0 else 0.0,
        std_interval_ms=np.std(intervals) if len(intervals) > 0 else 0.0,
        min_interval_ms=np.min(intervals) if len(intervals) > 0 else 0.0,
        max_interval_ms=np.max(intervals) if len(intervals) > 0 else 0.0,
        mean_jitter_ms=mean_jitter_ms,
        std_jitter_ms=std_jitter_ms,
        max_jitter_ms=max_jitter_ms,
        jitter_threshold_ms=jitter_threshold_ms,
        jitter_violations=int(jitter_violations),
        jitter_pass=jitter_pass,
        gaps=gaps,
        num_gaps=len(gaps),
        gap_threshold_ms=gap_threshold_ms,
        within_spec=True,  # Will be updated by validation
        issues=[],  # Will be updated by validation
        timestamps=timestamps,
        intervals=intervals,
        jitter=jitter
    )
    
    # Validate against spec
    within_spec, issues = validate_against_spec(
        result, 
        analysis_config.get('rate_tolerance_percent', 10.0)
    )
    result.within_spec = within_spec
    result.issues = issues
    
    return result


def analyze_experiment(sensor_data: Dict[str, pd.DataFrame],
                      sensor_specs: Dict[str, Any]) -> Dict[str, TimestampAnalysisResult]:
    """
    Analyze timestamps for all sensors in an experiment.
    
    Args:
        sensor_data: Dictionary mapping sensor names to DataFrames
        sensor_specs: Sensor specifications
        
    Returns:
        Dictionary mapping sensor names to analysis results
    """
    results = {}
    analysis_config = sensor_specs.get('analysis', {})
    
    for sensor_name, df in sensor_data.items():
        if 'time_from_sync' not in df.columns:
            warnings.warn(f"No timestamp column found for {sensor_name}")
            continue
        
        # Get timestamps as numpy array
        timestamps = df['time_from_sync'].values
        
        # Get sensor configuration
        from .data_loader import get_sensor_config
        sensor_config = get_sensor_config(sensor_name, sensor_specs)
        
        # Analyze timestamps
        result = analyze_timestamps(
            timestamps, 
            sensor_name, 
            sensor_config,
            analysis_config
        )
        
        results[sensor_name] = result
    
    return results


def compare_sensor_alignment(results: Dict[str, TimestampAnalysisResult]) -> Dict[str, Any]:
    """
    Compare timestamp alignment between sensors.
    
    Args:
        results: Dictionary of analysis results for each sensor
        
    Returns:
        Dictionary with cross-sensor alignment metrics
    """
    alignment_info = {
        'sensor_pairs': [],
        'reference_sensor': None,
        'max_offset_ms': 0.0
    }
    
    if len(results) < 2:
        return alignment_info
    
    # Find sensor with highest rate as reference
    max_rate = 0
    for name, result in results.items():
        if result.actual_rate_hz > max_rate:
            max_rate = result.actual_rate_hz
            alignment_info['reference_sensor'] = name
    
    # Compare start/end times between sensors
    for name1, result1 in results.items():
        for name2, result2 in results.items():
            if name1 >= name2:  # Avoid duplicates and self-comparison
                continue
            
            if len(result1.timestamps) > 0 and len(result2.timestamps) > 0:
                start_offset_ms = abs(result1.timestamps[0] - result2.timestamps[0]) * 1000
                end_offset_ms = abs(result1.timestamps[-1] - result2.timestamps[-1]) * 1000
                
                pair_info = {
                    'sensor1': name1,
                    'sensor2': name2,
                    'start_offset_ms': start_offset_ms,
                    'end_offset_ms': end_offset_ms,
                    'max_offset_ms': max(start_offset_ms, end_offset_ms)
                }
                
                alignment_info['sensor_pairs'].append(pair_info)
                alignment_info['max_offset_ms'] = max(
                    alignment_info['max_offset_ms'],
                    pair_info['max_offset_ms']
                )
    
    return alignment_info