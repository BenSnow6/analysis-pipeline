"""
Test suite for WP-4 multi-sensor fusion module.

Tests fusion logic, interpolation, and quality gating.
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path

from src.analysis.rpm.tracking import RPMFrame, RPMTimeSeries
from src.analysis.rpm.fusion import (
    select_best_sensor,
    compute_sensor_agreement,
    fuse_sensors_snr,
    interpolate_missing_frames,
    apply_median_filter
)


class TestSensorSelection:
    """Test sensor selection logic."""
    
    def test_select_best_sensor_by_snr(self):
        """Test that sensor with highest SNR is selected."""
        frames = [
            RPMFrame(time=1.0, rpm=1800, snr_db=10, sensor_id='S1', method='welch'),
            RPMFrame(time=1.0, rpm=1810, snr_db=15, sensor_id='S2', method='welch'),
            RPMFrame(time=1.0, rpm=1805, snr_db=12, sensor_id='S3', method='welch')
        ]
        
        best = select_best_sensor(frames, 1.0)
        assert best is not None
        assert best.sensor_id == 'S2'
        assert best.snr_db == 15
    
    def test_select_best_sensor_invalid_excluded(self):
        """Test that invalid sensors are excluded."""
        frames = [
            RPMFrame(time=1.0, rpm=1800, snr_db=8, sensor_id='S1', method='welch'),  # Below threshold
            RPMFrame(time=1.0, rpm=1810, snr_db=12, sensor_id='S2', method='welch'),
            RPMFrame(time=1.0, rpm=np.nan, snr_db=15, sensor_id='S3', method='welch')  # Invalid RPM
        ]
        
        best = select_best_sensor(frames, 1.0)
        assert best is not None
        assert best.sensor_id == 'S2'
    
    def test_select_best_sensor_no_valid(self):
        """Test handling when no sensors are valid."""
        frames = [
            RPMFrame(time=1.0, rpm=1800, snr_db=5, sensor_id='S1', method='welch'),
            RPMFrame(time=1.0, rpm=np.nan, snr_db=15, sensor_id='S2', method='welch')
        ]
        
        best = select_best_sensor(frames, 1.0)
        assert best is None


class TestSensorAgreement:
    """Test sensor agreement calculation."""
    
    def test_perfect_agreement(self):
        """Test perfect agreement between sensors."""
        frames = [
            RPMFrame(time=1.0, rpm=1800, snr_db=15, sensor_id='S1', method='welch'),
            RPMFrame(time=1.0, rpm=1800, snr_db=12, sensor_id='S2', method='welch'),
            RPMFrame(time=1.0, rpm=1800, snr_db=14, sensor_id='S3', method='welch')
        ]
        
        agreement = compute_sensor_agreement(frames)
        assert agreement == 1.0
    
    def test_partial_agreement(self):
        """Test partial agreement with some deviation."""
        frames = [
            RPMFrame(time=1.0, rpm=1800, snr_db=15, sensor_id='S1', method='welch'),
            RPMFrame(time=1.0, rpm=1820, snr_db=12, sensor_id='S2', method='welch'),
            RPMFrame(time=1.0, rpm=1810, snr_db=14, sensor_id='S3', method='welch')
        ]
        
        agreement = compute_sensor_agreement(frames, max_deviation=50.0)
        assert 0.5 < agreement < 1.0
    
    def test_poor_agreement(self):
        """Test poor agreement with large deviation."""
        frames = [
            RPMFrame(time=1.0, rpm=1800, snr_db=15, sensor_id='S1', method='welch'),
            RPMFrame(time=1.0, rpm=1900, snr_db=12, sensor_id='S2', method='welch'),
            RPMFrame(time=1.0, rpm=1700, snr_db=14, sensor_id='S3', method='welch')
        ]
        
        agreement = compute_sensor_agreement(frames, max_deviation=50.0)
        assert agreement <= 0.2


class TestMultiSensorFusion:
    """Test complete fusion pipeline."""
    
    def create_test_series(self, sensor_id: str, base_rpm: float, 
                          noise_level: float, snr_db: float) -> RPMTimeSeries:
        """Create a test RPM time series with noise."""
        times = np.arange(0, 10, 0.25)  # 10 seconds at 4 Hz
        rpms = base_rpm + noise_level * np.random.randn(len(times))
        
        frames = []
        for t, rpm in zip(times, rpms):
            frame = RPMFrame(
                time=float(t),
                rpm=float(rpm),
                snr_db=snr_db + np.random.randn(),  # Some SNR variation
                sensor_id=sensor_id,
                method='stft'
            )
            frames.append(frame)
        
        return RPMTimeSeries(
            frames=frames,
            experiment='test_exp',
            session='test',
            sensor_id=sensor_id
        )
    
    def test_fusion_basic(self):
        """Test basic fusion of multiple sensors."""
        # Create three sensors with different characteristics
        sensor_data = {
            'S1': self.create_test_series('S1', 1800, 10, 15),  # High SNR
            'S2': self.create_test_series('S2', 1810, 20, 12),  # Medium SNR
            'S3': self.create_test_series('S3', 1805, 30, 8)    # Low SNR
        }
        
        config = {
            'snr_thresh_db': 10,
            'fusion': {'strategy': 'snr_max'}
        }
        
        fused = fuse_sensors_snr(sensor_data, config)
        
        # Check properties
        assert fused is not None
        assert len(fused.frames) == len(sensor_data['S1'].frames)
        assert fused.sensor_id == 'fused'
        
        # Should prefer S1 (highest SNR)
        s1_count = sum(1 for f in fused.frames if 'S1' in f.sensor_id)
        assert s1_count > len(fused.frames) * 0.8  # Most frames from S1
    
    def test_fusion_with_gaps(self):
        """Test fusion with sensor dropouts."""
        # Create series with gaps
        s1 = self.create_test_series('S1', 1800, 10, 15)
        s2 = self.create_test_series('S2', 1810, 20, 12)
        
        # Remove some frames from S1
        s1.frames = [f for f in s1.frames if f.time < 5 or f.time > 7]
        
        sensor_data = {'S1': s1, 'S2': s2}
        config = {'snr_thresh_db': 10}
        
        fused = fuse_sensors_snr(sensor_data, config)
        
        # Should use S2 during S1 gap
        gap_frames = [f for f in fused.frames if 5 <= f.time <= 7]
        s2_gap_count = sum(1 for f in gap_frames if 'S2' in f.sensor_id)
        assert s2_gap_count == len(gap_frames)


class TestInterpolation:
    """Test interpolation functionality."""
    
    def test_interpolate_small_gaps(self):
        """Test interpolation of small gaps."""
        # Create series with gaps
        times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Include gap times
        frames = []
        
        for t in times:
            if t in [4, 5]:  # Create gap
                frame = RPMFrame(time=t, rpm=np.nan, snr_db=0, 
                               sensor_id='test', method='welch')
            else:
                frame = RPMFrame(time=t, rpm=1800 + t*10, snr_db=15,
                               sensor_id='test', method='welch')
            frames.append(frame)
        
        series = RPMTimeSeries(frames, 'test', 'test', 'test')
        
        # Interpolate with 5s max gap
        interpolated = interpolate_missing_frames(series, max_gap_s=5.0)
        
        # Check that gap was filled
        assert interpolated.availability > series.availability
        
        # Check interpolated values are reasonable
        for frame in interpolated.frames:
            if frame.method == 'interpolated':
                assert 1830 <= frame.rpm <= 1860  # Between surrounding values
    
    def test_no_interpolate_large_gaps(self):
        """Test that large gaps are not interpolated."""
        # Create series with large gap
        times = list(range(0, 5)) + list(range(15, 20))  # 10s gap
        frames = []
        
        for t in times:
            frame = RPMFrame(time=t, rpm=1800, snr_db=15,
                           sensor_id='test', method='welch')
            frames.append(frame)
        
        # Add gap frames
        for t in range(5, 15):
            frame = RPMFrame(time=t, rpm=np.nan, snr_db=0,
                           sensor_id='test', method='welch')
            frames.append(frame)
        
        series = RPMTimeSeries(sorted(frames, key=lambda f: f.time), 
                             'test', 'test', 'test')
        
        # Interpolate with 5s max gap
        interpolated = interpolate_missing_frames(series, max_gap_s=5.0)
        
        # Large gap should not be filled
        gap_frames = [f for f in interpolated.frames if 5 <= f.time < 15]
        valid_gap_frames = [f for f in gap_frames if f.is_valid()]
        assert len(valid_gap_frames) == 0


class TestMedianFilter:
    """Test median filtering for outlier removal."""
    
    def test_remove_outliers(self):
        """Test that outliers are removed by median filter."""
        # Create series with outliers
        times = np.arange(0, 10, 0.25)
        frames = []
        
        for i, t in enumerate(times):
            if i in [10, 20, 30]:  # Add outliers
                rpm = 2500  # Way off from baseline
            else:
                rpm = 1800 + np.random.randn() * 5
            
            frame = RPMFrame(time=t, rpm=rpm, snr_db=15,
                           sensor_id='test', method='welch')
            frames.append(frame)
        
        series = RPMTimeSeries(frames, 'test', 'test', 'test')
        
        # Apply median filter
        filtered = apply_median_filter(series, window_s=1.0)
        
        # Check outliers were reduced
        filtered_rpms = [f.rpm for f in filtered.frames if f.is_valid()]
        assert all(1700 < rpm < 1900 for rpm in filtered_rpms)
    
    def test_preserve_trends(self):
        """Test that median filter preserves trends."""
        # Create series with linear trend
        times = np.arange(0, 10, 0.25)
        frames = []
        
        for t in times:
            rpm = 1800 + t * 20  # Linear increase
            frame = RPMFrame(time=t, rpm=rpm, snr_db=15,
                           sensor_id='test', method='welch')
            frames.append(frame)
        
        series = RPMTimeSeries(frames, 'test', 'test', 'test')
        
        # Apply median filter
        filtered = apply_median_filter(series, window_s=0.5)
        
        # Check trend is preserved
        rpms = [f.rpm for f in filtered.frames]
        # Should still be increasing
        assert all(rpms[i] <= rpms[i+1] for i in range(len(rpms)-1))