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
    """
    time: float
    rpm: float
    snr_db: float
    sensor_id: str
    method: Literal['welch', 'stft', 'order_tracking']
    harmonics: Optional[dict] = field(default_factory=dict)
    confidence: Optional[float] = None
    
    def is_valid(self, snr_threshold: float = 10.0) -> bool:
        """
        Check if estimate meets confidence threshold.
        
        Args:
            snr_threshold: Minimum SNR in dB for valid estimate
            
        Returns:
            True if estimate is valid, False otherwise
        """
        return self.snr_db >= snr_threshold
    
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
        experiment: Experiment identifier
        session: Morning or afternoon session
        sensor_id: Source sensor identifier
    """
    frames: List[RPMFrame]
    experiment: str
    session: Literal['morning', 'afternoon']
    sensor_id: str
    
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