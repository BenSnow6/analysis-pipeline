"""
RPM tracking data structures and utilities.

This module defines the core data structures for storing and managing
RPM estimates from vibration analysis.
"""

from dataclasses import dataclass, field
from typing import Literal, List, Optional
import numpy as np


@dataclass
class RPMFrame:
    """
    Container for a single RPM estimate with metadata.
    
    Attributes:
        time: Timestamp in seconds
        rpm: Estimated RPM value
        snr_db: Signal-to-noise ratio in decibels
        sensor_id: Identifier of the source sensor
        method: Estimation method used
        harmonics: Optional dictionary of harmonic amplitudes
        confidence: Optional confidence score (0-1)
        metadata: Optional dictionary for additional information
    """
    time: float
    rpm: float
    snr_db: float
    sensor_id: str
    method: Literal['welch', 'stft', 'order_tracking']
    harmonics: Optional[dict] = field(default_factory=dict)
    confidence: Optional[float] = None
    metadata: Optional[dict] = field(default_factory=dict)
    
    def is_valid(self, snr_threshold: float = 10.0) -> bool:
        """
        Check if estimate meets confidence threshold.
        
        Args:
            snr_threshold: Minimum SNR in dB for valid estimate
            
        Returns:
            True if estimate is valid, False otherwise
        """
        # Check for NaN RPM values and SNR threshold
        return not np.isnan(self.rpm) and self.snr_db >= snr_threshold
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if self.rpm < 0:
            raise ValueError(f"RPM must be non-negative, got {self.rpm}")
        if not 0 <= self.time:
            raise ValueError(f"Time must be non-negative, got {self.time}")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class RPMTimeSeries:
    """
    Container for a time series of RPM estimates.
    
    Attributes:
        frames: List of RPMFrame objects
        experiment: Optional experiment identifier
        session: Optional morning or afternoon session
        sensor_id: Optional source sensor identifier
        metadata: Optional dictionary for additional information
    """
    frames: List[RPMFrame]
    experiment: Optional[str] = None
    session: Optional[Literal['morning', 'afternoon']] = None
    sensor_id: Optional[str] = None
    metadata: Optional[dict] = field(default_factory=dict)
    
    def get_valid_frames(self, snr_threshold: float = 10.0) -> List[RPMFrame]:
        """Return only frames that meet the SNR threshold."""
        return [f for f in self.frames if f.is_valid(snr_threshold)]
    
    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert to numpy arrays for analysis.
        
        Returns:
            Tuple of (times, rpms, snrs) as numpy arrays
        """
        times = np.array([f.time for f in self.frames])
        rpms = np.array([f.rpm for f in self.frames])
        snrs = np.array([f.snr_db for f in self.frames])
        return times, rpms, snrs
    
    @property
    def duration(self) -> float:
        """Total duration of the time series in seconds."""
        if not self.frames:
            return 0.0
        return self.frames[-1].time - self.frames[0].time
    
    @property
    def mean_rpm(self) -> float:
        """Mean RPM across all valid frames."""
        valid_frames = self.get_valid_frames()
        if not valid_frames:
            return np.nan
        return np.mean([f.rpm for f in valid_frames])
    
    @property
    def availability(self) -> float:
        """Percentage of valid frames."""
        if not self.frames:
            return 0.0
        valid_count = len(self.get_valid_frames())
        return 100.0 * valid_count / len(self.frames)


def smooth_rpm_series(time: np.ndarray, rpm: np.ndarray, 
                     method: str = 'polynomial', 
                     window: int = 5,
                     high_rate_threshold: float = 150.0) -> np.ndarray:
    """
    Apply lightweight smoothing to RPM series.
    
    This function applies smoothing only to regions with high rate of change
    to preserve accuracy in steady-state regions while reducing noise in
    transient regions.
    
    Args:
        time: Time array in seconds
        rpm: RPM values (may contain NaN)
        method: Smoothing method
            - 'polynomial': Fit low-order polynomial in sliding window
            - 'median': Median filter with outlier rejection
            - 'moving_avg': Weighted moving average
        window: Window size in samples
        high_rate_threshold: RPM/s threshold for applying smoothing
        
    Returns:
        Smoothed RPM array (same size as input)
    """
    # Handle NaN values
    valid_mask = ~np.isnan(rpm)
    if not np.any(valid_mask):
        return rpm.copy()
    
    # Calculate rate of change
    dt = np.diff(time)
    if len(dt) == 0 or np.any(dt <= 0):
        return rpm.copy()
    
    # Compute RPM rate of change (RPM/s)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return rpm.copy()
    
    # Get differences for valid points
    rpm_valid = rpm[valid_mask]
    time_valid = time[valid_mask]
    dt_valid = np.diff(time_valid)
    
    if len(dt_valid) == 0 or np.any(dt_valid <= 0):
        return rpm.copy()
        
    rpm_rate = np.abs(np.diff(rpm_valid) / dt_valid)
    
    # Identify high-rate regions
    high_rate_mask = np.zeros(len(rpm), dtype=bool)
    high_rate_indices_in_valid = np.where(rpm_rate > high_rate_threshold)[0]
    
    # Map back to original indices
    for idx in high_rate_indices_in_valid:
        # Get the original index range
        if idx < len(valid_indices) - 1:
            orig_idx_start = valid_indices[idx]
            orig_idx_end = valid_indices[idx + 1]
            
            # Expand high-rate regions by window/2 on each side
            half_window = window // 2
            start = max(0, orig_idx_start - half_window)
            end = min(len(rpm), orig_idx_end + half_window)
            high_rate_mask[start:end] = True
    
    # Apply smoothing based on method
    smoothed_rpm = rpm.copy()
    
    if method == 'polynomial':
        smoothed_rpm = _polynomial_smooth(time, rpm, valid_mask, 
                                        high_rate_mask, window)
    elif method == 'median':
        smoothed_rpm = _median_smooth(rpm, valid_mask, 
                                    high_rate_mask, window)
    elif method == 'moving_avg':
        smoothed_rpm = _moving_avg_smooth(rpm, valid_mask, 
                                        high_rate_mask, window)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed_rpm


def _polynomial_smooth(time: np.ndarray, rpm: np.ndarray, 
                      valid_mask: np.ndarray, high_rate_mask: np.ndarray,
                      window: int, poly_order: int = 2) -> np.ndarray:
    """Apply polynomial smoothing to high-rate regions."""
    from scipy.signal import savgol_filter
    
    smoothed = rpm.copy()
    
    # Find continuous high-rate regions
    regions = _find_continuous_regions(high_rate_mask & valid_mask)
    
    for start, end in regions:
        if end - start < window:
            continue
            
        # Extract region data
        region_time = time[start:end]
        region_rpm = rpm[start:end]
        region_valid = valid_mask[start:end]
        
        if np.sum(region_valid) < window:
            continue
        
        # Apply Savitzky-Golay filter (polynomial smoothing)
        try:
            smoothed_region = savgol_filter(
                region_rpm[region_valid], 
                window_length=min(window, np.sum(region_valid)),
                polyorder=min(poly_order, np.sum(region_valid) - 1),
                mode='nearest'
            )
            smoothed[start:end][region_valid] = smoothed_region
        except Exception:
            # If smoothing fails, keep original values
            pass
    
    return smoothed


def _median_smooth(rpm: np.ndarray, valid_mask: np.ndarray,
                  high_rate_mask: np.ndarray, window: int) -> np.ndarray:
    """Apply median filter to high-rate regions."""
    from scipy.signal import medfilt
    
    smoothed = rpm.copy()
    
    # Apply median filter only to high-rate regions
    if np.any(high_rate_mask & valid_mask):
        # Create temporary array for filtering
        temp_rpm = rpm.copy()
        temp_rpm[~valid_mask] = np.nan
        
        # Apply median filter
        filtered = medfilt(temp_rpm, kernel_size=window)
        
        # Update only high-rate regions
        update_mask = high_rate_mask & valid_mask
        smoothed[update_mask] = filtered[update_mask]
    
    return smoothed


def _moving_avg_smooth(rpm: np.ndarray, valid_mask: np.ndarray,
                      high_rate_mask: np.ndarray, window: int) -> np.ndarray:
    """Apply weighted moving average to high-rate regions."""
    smoothed = rpm.copy()
    
    # Create weights (higher weight for center)
    weights = np.hanning(window)
    weights /= weights.sum()
    
    # Apply to each high-rate point
    for i in range(len(rpm)):
        if not (high_rate_mask[i] and valid_mask[i]):
            continue
            
        # Get window bounds
        start = max(0, i - window // 2)
        end = min(len(rpm), i + window // 2 + 1)
        
        # Extract window data
        window_rpm = rpm[start:end]
        window_valid = valid_mask[start:end]
        
        if np.sum(window_valid) > 0:
            # Apply weighted average only to valid points
            valid_rpm = window_rpm[window_valid]
            # Adjust weights for valid points only
            valid_weights = weights[:len(valid_rpm)]
            valid_weights /= valid_weights.sum()
            
            smoothed[i] = np.sum(valid_rpm * valid_weights)
    
    return smoothed


def _find_continuous_regions(mask: np.ndarray) -> List[tuple[int, int]]:
    """Find continuous True regions in a boolean mask."""
    regions = []
    start = None
    
    for i in range(len(mask)):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            regions.append((start, i))
            start = None
    
    if start is not None:
        regions.append((start, len(mask)))
    
    return regions