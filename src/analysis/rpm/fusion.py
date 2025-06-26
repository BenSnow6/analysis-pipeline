"""
Multi-sensor fusion for RPM estimation.

This module implements sensor fusion strategies for combining
RPM estimates from multiple IMUs.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from .tracking import RPMFrame, RPMTimeSeries

logger = logging.getLogger(__name__)


def select_best_sensor(frames: List[RPMFrame], time: float) -> Optional[RPMFrame]:
    """
    Select the best sensor estimate for a given time based on SNR.
    
    Args:
        frames: List of RPMFrame objects from different sensors
        time: Target time
        
    Returns:
        Best RPMFrame or None if no valid estimates
    """
    # Filter frames at the target time
    time_frames = [f for f in frames if abs(f.time - time) < 0.01]
    
    if not time_frames:
        return None
    
    # Filter by validity
    valid_frames = [f for f in time_frames if f.is_valid()]
    
    if not valid_frames:
        return None
    
    # Select frame with highest SNR
    best_frame = max(valid_frames, key=lambda f: f.snr_db)
    
    return best_frame


def compute_sensor_agreement(frames: List[RPMFrame], max_deviation: float = 50.0) -> float:
    """
    Compute agreement score between sensors.
    
    Args:
        frames: List of RPMFrame objects from different sensors
        max_deviation: Maximum acceptable RPM deviation
        
    Returns:
        Agreement score (0-1)
    """
    if len(frames) < 2:
        return 1.0
    
    rpms = [f.rpm for f in frames if f.is_valid()]
    
    if len(rpms) < 2:
        return 1.0
    
    # Compute standard deviation
    std_rpm = np.std(rpms)
    
    # Convert to agreement score
    agreement = max(0, 1 - std_rpm / max_deviation)
    
    return agreement


def fuse_sensors_snr(sensor_series: Dict[str, RPMTimeSeries], 
                    config: dict) -> RPMTimeSeries:
    """
    Fuse multiple sensor estimates using SNR-based selection.
    
    Args:
        sensor_series: Dictionary mapping sensor ID to time series
        config: Configuration dictionary
        
    Returns:
        Fused RPMTimeSeries
    """
    # Get unique timestamps across all sensors
    all_times = set()
    for series in sensor_series.values():
        all_times.update([f.time for f in series.frames])
    
    all_times = sorted(all_times)
    
    # Fuse at each timestamp
    fused_frames = []
    
    for t in all_times:
        # Collect frames from all sensors at this time
        frames_at_t = []
        for sensor_id, series in sensor_series.items():
            # Find frame at time t
            for frame in series.frames:
                if abs(frame.time - t) < 0.01:
                    frames_at_t.append(frame)
                    break
        
        # Select best sensor
        best_frame = select_best_sensor(frames_at_t, t)
        
        if best_frame:
            # Create fused frame
            agreement = compute_sensor_agreement(frames_at_t)
            
            fused_frame = RPMFrame(
                time=t,
                rpm=best_frame.rpm,
                snr_db=best_frame.snr_db,
                sensor_id=f"fused_{best_frame.sensor_id}",
                method=best_frame.method,
                confidence=agreement
            )
            fused_frames.append(fused_frame)
    
    # Create fused time series
    experiment = list(sensor_series.values())[0].experiment
    session = list(sensor_series.values())[0].session
    
    fused_series = RPMTimeSeries(
        frames=fused_frames,
        experiment=experiment,
        session=session,
        sensor_id="fused"
    )
    
    logger.info(f"Fused {len(sensor_series)} sensors: "
                f"{fused_series.availability:.1f}% availability")
    
    return fused_series


def interpolate_missing_frames(series: RPMTimeSeries, 
                             max_gap_s: float = 5.0) -> RPMTimeSeries:
    """
    Interpolate missing frames in RPM time series.
    
    Args:
        series: Input RPM time series
        max_gap_s: Maximum gap to interpolate (seconds)
        
    Returns:
        Interpolated time series
    """
    if len(series.frames) < 2:
        return series
    
    times, rpms, snrs = series.to_arrays()
    
    # Find valid frames
    valid_mask = np.array([f.is_valid() for f in series.frames])
    
    if np.sum(valid_mask) < 2:
        return series
    
    # Interpolate RPM values
    from scipy import interpolate
    
    valid_times = times[valid_mask]
    valid_rpms = rpms[valid_mask]
    
    # Create interpolator
    f_interp = interpolate.interp1d(valid_times, valid_rpms, 
                                   kind='linear', bounds_error=False,
                                   fill_value='extrapolate')
    
    # Interpolate at all times
    interp_rpms = f_interp(times)
    
    # Create new frames
    new_frames = []
    for i, frame in enumerate(series.frames):
        if frame.is_valid():
            new_frames.append(frame)
        else:
            # Check gap size
            prev_valid = np.where(valid_times < times[i])[0]
            next_valid = np.where(valid_times > times[i])[0]
            
            gap_ok = True
            if len(prev_valid) > 0 and len(next_valid) > 0:
                gap = valid_times[next_valid[0]] - valid_times[prev_valid[-1]]
                gap_ok = gap <= max_gap_s
            
            if gap_ok and not np.isnan(interp_rpms[i]):
                # Create interpolated frame
                # Use minimum valid SNR to ensure frame is valid
                interp_frame = RPMFrame(
                    time=frame.time,
                    rpm=float(interp_rpms[i]),
                    snr_db=10.0,  # Minimum valid SNR for interpolated frames
                    sensor_id=frame.sensor_id,
                    method='interpolated',
                    confidence=0.5
                )
                new_frames.append(interp_frame)
            else:
                new_frames.append(frame)
    
    # Create new series
    interp_series = RPMTimeSeries(
        frames=new_frames,
        experiment=series.experiment,
        session=series.session,
        sensor_id=series.sensor_id
    )
    
    logger.info(f"Interpolation: {series.availability:.1f}% -> "
                f"{interp_series.availability:.1f}% availability")
    
    return interp_series


def apply_median_filter(series: RPMTimeSeries, window_s: float = 1.0) -> RPMTimeSeries:
    """
    Apply median filter to remove outliers.
    
    Args:
        series: Input RPM time series
        window_s: Window size in seconds
        
    Returns:
        Filtered time series
    """
    times, rpms, snrs = series.to_arrays()
    
    # Calculate window size in samples
    dt = np.median(np.diff(times))
    window_samples = int(window_s / dt)
    
    if window_samples < 3:
        window_samples = 3
    
    # Apply median filter to valid RPMs
    from scipy.ndimage import median_filter
    
    valid_mask = np.array([f.is_valid() for f in series.frames])
    filtered_rpms = rpms.copy()
    
    if np.sum(valid_mask) > window_samples:
        filtered_rpms[valid_mask] = median_filter(rpms[valid_mask], size=window_samples)
    
    # Create filtered frames
    filtered_frames = []
    for i, frame in enumerate(series.frames):
        if frame.is_valid():
            filtered_frame = RPMFrame(
                time=frame.time,
                rpm=float(filtered_rpms[i]),
                snr_db=frame.snr_db,
                sensor_id=frame.sensor_id,
                method=frame.method,
                confidence=frame.confidence
            )
            filtered_frames.append(filtered_frame)
        else:
            filtered_frames.append(frame)
    
    # Create filtered series
    filtered_series = RPMTimeSeries(
        frames=filtered_frames,
        experiment=series.experiment,
        session=series.session,
        sensor_id=series.sensor_id
    )
    
    return filtered_series