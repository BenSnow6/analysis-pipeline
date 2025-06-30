"""
Tests for RPM tracking data structures.
"""

import pytest
import numpy as np
from src.analysis.rpm.tracking import RPMFrame, RPMTimeSeries


class TestRPMFrame:
    """Test RPMFrame dataclass."""
    
    def test_basic_instantiation(self):
        """Test creating a basic RPMFrame."""
        frame = RPMFrame(
            time=10.0,
            rpm=1800.0,
            snr_db=15.0,
            sensor_id='Sensor_3',
            method='welch'
        )
        
        assert frame.time == 10.0
        assert frame.rpm == 1800.0
        assert frame.snr_db == 15.0
        assert frame.sensor_id == 'Sensor_3'
        assert frame.method == 'welch'
        assert frame.harmonics == {}
        assert frame.confidence is None
    
    def test_is_valid_method(self):
        """Test the is_valid() method."""
        # Valid frame (SNR > 10)
        frame1 = RPMFrame(
            time=1.0, rpm=1500.0, snr_db=12.0, 
            sensor_id='test', method='welch'
        )
        assert frame1.is_valid() is True
        assert frame1.is_valid(snr_threshold=10.0) is True
        assert frame1.is_valid(snr_threshold=15.0) is False
        
        # Invalid frame (SNR < 10)
        frame2 = RPMFrame(
            time=2.0, rpm=1600.0, snr_db=8.0,
            sensor_id='test', method='welch'
        )
        assert frame2.is_valid() is False
        assert frame2.is_valid(snr_threshold=5.0) is True
    
    def test_with_harmonics(self):
        """Test RPMFrame with harmonics data."""
        harmonics = {1: 0.5, 2: 0.3, 3: 0.1}
        frame = RPMFrame(
            time=5.0, rpm=2000.0, snr_db=20.0,
            sensor_id='Sensor_4', method='stft',
            harmonics=harmonics,
            confidence=0.95
        )
        
        assert frame.harmonics == harmonics
        assert frame.confidence == 0.95
    
    def test_validation_errors(self):
        """Test validation of invalid inputs."""
        # Negative RPM
        with pytest.raises(ValueError, match="RPM must be non-negative"):
            RPMFrame(
                time=1.0, rpm=-100.0, snr_db=10.0,
                sensor_id='test', method='welch'
            )
        
        # Negative time
        with pytest.raises(ValueError, match="Time must be non-negative"):
            RPMFrame(
                time=-1.0, rpm=1500.0, snr_db=10.0,
                sensor_id='test', method='welch'
            )
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between"):
            RPMFrame(
                time=1.0, rpm=1500.0, snr_db=10.0,
                sensor_id='test', method='welch',
                confidence=1.5
            )


class TestRPMTimeSeries:
    """Test RPMTimeSeries dataclass."""
    
    def create_test_series(self):
        """Create a test time series."""
        frames = [
            RPMFrame(time=0.0, rpm=1000.0, snr_db=15.0, sensor_id='test', method='welch'),
            RPMFrame(time=1.0, rpm=1100.0, snr_db=12.0, sensor_id='test', method='welch'),
            RPMFrame(time=2.0, rpm=1200.0, snr_db=8.0, sensor_id='test', method='welch'),  # Invalid
            RPMFrame(time=3.0, rpm=1300.0, snr_db=18.0, sensor_id='test', method='welch'),
            RPMFrame(time=4.0, rpm=1400.0, snr_db=5.0, sensor_id='test', method='welch'),   # Invalid
        ]
        
        return RPMTimeSeries(
            frames=frames,
            experiment='test_experiment',
            session='afternoon',
            sensor_id='test'
        )
    
    def test_basic_properties(self):
        """Test basic properties of time series."""
        series = self.create_test_series()
        
        assert len(series.frames) == 5
        assert series.experiment == 'test_experiment'
        assert series.session == 'afternoon'
        assert series.sensor_id == 'test'
        assert series.duration == 4.0
    
    def test_get_valid_frames(self):
        """Test filtering valid frames."""
        series = self.create_test_series()
        
        # Default threshold (10 dB)
        valid_frames = series.get_valid_frames()
        assert len(valid_frames) == 3
        assert all(f.snr_db >= 10.0 for f in valid_frames)
        
        # Custom threshold
        valid_frames_15 = series.get_valid_frames(snr_threshold=15.0)
        assert len(valid_frames_15) == 2
    
    def test_to_arrays(self):
        """Test conversion to numpy arrays."""
        series = self.create_test_series()
        times, rpms, snrs = series.to_arrays()
        
        assert isinstance(times, np.ndarray)
        assert isinstance(rpms, np.ndarray)
        assert isinstance(snrs, np.ndarray)
        
        assert len(times) == 5
        assert times[0] == 0.0
        assert times[-1] == 4.0
        
        assert rpms[0] == 1000.0
        assert rpms[2] == 1200.0
        
        assert snrs[1] == 12.0
        assert snrs[3] == 18.0
    
    def test_statistics(self):
        """Test statistical properties."""
        series = self.create_test_series()
        
        # Mean RPM (only valid frames)
        assert series.mean_rpm == pytest.approx(1133.33, rel=0.01)
        
        # Availability
        assert series.availability == 60.0  # 3 out of 5 frames valid
    
    def test_empty_series(self):
        """Test empty time series handling."""
        empty_series = RPMTimeSeries(
            frames=[],
            experiment='empty',
            session='morning',
            sensor_id='none'
        )
        
        assert empty_series.duration == 0.0
        assert np.isnan(empty_series.mean_rpm)
        assert empty_series.availability == 0.0
        
        times, rpms, snrs = empty_series.to_arrays()
        assert len(times) == 0
        assert len(rpms) == 0
        assert len(snrs) == 0