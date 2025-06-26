"""
Tests for quality assessment module.
"""

import numpy as np
import pytest
from pathlib import Path
import sys
import json

# Import quality module
from src.analysis.rpm.quality import (
    process_quality_windows, compute_window_metrics, assess_signal_quality,
    classify_overall_quality, check_multi_axis_quality, validate_time_alignment
)


class TestWindowProcessing:
    """Test window processing with different handling strategies."""
    
    def test_drop_strategy(self):
        """Test drop strategy for partial windows."""
        data = np.arange(95)  # Not divisible by 30
        windows = process_quality_windows(data, 30, "drop")
        
        # Should have only 3 complete windows
        assert len(windows) == 3
        assert all(not w['is_partial'] for w in windows)
    
    def test_pad_strategy(self):
        """Test pad strategy for partial windows."""
        data = np.arange(95)
        windows = process_quality_windows(data, 30, "pad")
        
        # Should have 4 windows
        assert len(windows) == 4
        # Last window should be padded
        assert len(windows[-1]['data']) == 30
        assert windows[-1]['is_partial'] == True
    
    def test_process_partial_strategy(self):
        """Test process_partial strategy."""
        data = np.arange(95)
        windows = process_quality_windows(data, 30, "process_partial")
        
        # Should have 4 windows
        assert len(windows) == 4
        # Last window should be shorter
        assert len(windows[-1]['data']) == 5
        assert windows[-1]['is_partial'] == True


class TestWindowMetrics:
    """Test computation of window metrics."""
    
    def test_basic_metrics(self):
        """Test basic metric calculations."""
        # Create known signal
        signal = np.ones(100) * 2.0
        metrics = compute_window_metrics(signal)
        
        assert metrics['mean'] == 2.0
        assert metrics['rms'] == 2.0
        assert metrics['peak'] == 2.0
        assert metrics['std'] == 0.0
    
    def test_clipping_detection(self):
        """Test clipping detection logic."""
        # Signal with clipping
        signal = np.random.randn(100) * 2
        signal[0:10] = 15.5  # Above 95% of 16
        
        metrics = compute_window_metrics(signal, max_value=16.0)
        
        assert metrics['clipping_detected'] == True
        assert metrics['clipping_samples'] >= 10
    
    def test_peak_to_rms(self):
        """Test peak-to-RMS ratio calculation."""
        # Sine wave has specific peak-to-RMS ratio
        t = np.linspace(0, 2*np.pi, 1000)
        signal = np.sin(t)
        
        metrics = compute_window_metrics(signal)
        
        # For sine wave, peak/RMS = sqrt(2) ≈ 1.414
        expected_ratio = np.sqrt(2)
        assert abs(metrics['peak_to_rms'] - expected_ratio) < 0.01


class TestSignalQualityAssessment:
    """Test overall signal quality assessment."""
    
    def test_quality_assessment(self):
        """Test complete quality assessment pipeline."""
        # Create test signal
        fs = 200
        duration = 120  # 2 minutes for 4 windows
        time = np.arange(0, duration, 1/fs)
        signal = np.random.randn(len(time))
        
        # Add some clipping to window 2
        window_size = int(30 * fs)
        signal[window_size:window_size+100] = 15.0
        
        # Test configuration
        config = {
            'fs': fs,
            'wp1': {
                'sensors': {'max_g_range': 16.0},
                'quality': {
                    'window_sec': 30.0,
                    'window_handling': 'process_partial',
                    'clipping_threshold': 0.95,
                    'thresholds': {
                        'excellent': 0.01,
                        'good': 0.05,
                        'fair': 0.10,
                        'poor': 1.0
                    }
                }
            }
        }
        
        results = assess_signal_quality(signal, time, config, 'test_sensor')
        
        # Check structure
        assert 'summary' in results
        assert 'windows' in results
        assert 'parameters_used' in results
        
        # Check summary
        assert results['summary']['total_windows'] == 4
        assert results['summary']['clipped_windows'] >= 1
        assert results['summary']['sensor_id'] == 'test_sensor'
    
    def test_quality_classification(self):
        """Test overall quality classification."""
        thresholds = {
            'excellent': 0.01,
            'good': 0.05,
            'fair': 0.10,
            'poor': 1.0
        }
        
        test_cases = [
            (0.005, 'excellent'),  # 0.5% clipped
            (0.03, 'good'),        # 3% clipped
            (0.08, 'fair'),        # 8% clipped
            (0.15, 'poor'),        # 15% clipped
        ]
        
        for clipping_ratio, expected in test_cases:
            quality = classify_overall_quality(clipping_ratio, thresholds)
            assert quality == expected


class TestMultiAxisQuality:
    """Test multi-axis quality checks."""
    
    def test_axis_quality_check(self):
        """Test quality check across three axes."""
        # Create test data
        config = {
            'wp1': {
                'sensors': {'max_g_range': 16.0}
            }
        }
        
        # Good signals
        x = np.random.randn(1000) * 0.5
        y = np.random.randn(1000) * 0.5
        z = np.random.randn(1000) * 0.5
        
        # Add issue to z-axis
        z += 3.0  # DC offset
        
        results = check_multi_axis_quality(x, y, z, config)
        
        assert results['x']['quality'] == 'good'
        assert results['y']['quality'] == 'good'
        assert results['z']['quality'] == 'poor'
        assert 'dc_offset' in results['z']['issues']


class TestTimeAlignment:
    """Test time vector validation."""
    
    def test_valid_time_vector(self):
        """Test validation of properly sampled time vector."""
        fs = 200
        time = np.arange(0, 10, 1/fs)
        
        is_valid, issues = validate_time_alignment(time, fs)
        
        assert is_valid == True
        assert len(issues) == 0
    
    def test_non_monotonic_time(self):
        """Test detection of non-monotonic time."""
        fs = 200
        time = np.arange(0, 10, 1/fs)
        time[500] = time[499]  # Duplicate timestamp
        
        is_valid, issues = validate_time_alignment(time, fs)
        
        assert is_valid == False
        assert any('monotonic' in issue for issue in issues)
    
    def test_sampling_rate_mismatch(self):
        """Test detection of sampling rate issues."""
        fs_expected = 200
        fs_actual = 190
        time = np.arange(0, 10, 1/fs_actual)
        
        is_valid, issues = validate_time_alignment(time, fs_expected)
        
        assert is_valid == False
        assert any('Sampling rate mismatch' in issue for issue in issues)
    
    def test_time_gaps(self):
        """Test detection of time gaps."""
        fs = 200
        time = np.arange(0, 10, 1/fs)
        # Create gap
        time[500:] += 0.1  # 100ms gap
        
        is_valid, issues = validate_time_alignment(time, fs)
        
        assert is_valid == False
        assert any('gaps' in issue for issue in issues)


def test_quality_report_generation():
    """Test quality report generation."""
    from quality import generate_quality_report
    
    # Mock quality results
    quality_results = {
        'summary': {
            'sensor_id': 'Sensor_3',
            'total_windows': 10,
            'clipped_windows': 1,
            'clipping_percentage': 10.0,
            'overall_quality': 'fair',
            'quality_score': 0.9,
            'duration_seconds': 300.0,
            'sample_count': 60000
        },
        'parameters_used': {
            'window_sec': 30.0,
            'clipping_threshold': 0.95
        },
        'windows': []
    }
    
    report = generate_quality_report(quality_results, 'test_exp', 'morning', '1.0')
    
    # Check report structure
    assert report['experiment'] == 'test_exp'
    assert report['session'] == 'morning'
    assert report['config_version'] == '1.0'
    assert 'processing_timestamp' in report
    assert 'processing_log' in report
    
    # Check processing log
    assert len(report['processing_log']['warnings']) > 0  # Should warn about 10% clipping


def test_quality_report_save(tmp_path):
    """Test saving quality report to JSON."""
    from quality import save_quality_report
    
    report = {
        'experiment': 'test',
        'session': 'morning',
        'summary': {'overall_quality': 'good'}
    }
    
    output_path = tmp_path / 'quality_report.json'
    save_quality_report(report, output_path)
    
    # Verify file exists and is valid JSON
    assert output_path.exists()
    with open(output_path) as f:
        loaded = json.load(f)
    assert loaded['experiment'] == 'test'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])