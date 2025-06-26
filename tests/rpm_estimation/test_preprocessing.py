"""
Tests for preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Import preprocessing module
from src.analysis.rpm.preprocess import (
    high_pass_filter, compute_vibration_magnitude, remove_gravity,
    apply_anti_alias_filter, compute_quality_metrics, detrend_signal
)


class TestHighPassFilter:
    """Test high-pass filtering functionality."""
    
    def test_dc_removal(self):
        """Test that DC offset is removed."""
        # Create signal with DC offset
        fs = 200
        t = np.arange(0, 10, 1/fs)
        dc_offset = 5.0
        signal = dc_offset + 0.1 * np.sin(2 * np.pi * 10 * t)
        
        # Apply high-pass filter
        filtered = high_pass_filter(signal, fs, cutoff=5.0)
        
        # Check mean is near zero
        assert abs(np.mean(filtered)) < 0.01, "DC offset not properly removed"
    
    def test_passband_signal(self):
        """Test that signals above cutoff pass through."""
        fs = 200
        t = np.arange(0, 10, 1/fs)
        
        # 20 Hz signal (well above 5 Hz cutoff)
        signal = np.sin(2 * np.pi * 20 * t)
        filtered = high_pass_filter(signal, fs, cutoff=5.0)
        
        # Signal should be mostly preserved
        # Allow for some attenuation but should be > 90% of original
        assert np.max(filtered) > 0.9 * np.max(signal)
    
    def test_stopband_signal(self):
        """Test that signals below cutoff are attenuated."""
        fs = 200
        t = np.arange(0, 10, 1/fs)
        
        # 1 Hz signal (well below 5 Hz cutoff)
        signal = np.sin(2 * np.pi * 1 * t)
        filtered = high_pass_filter(signal, fs, cutoff=5.0)
        
        # Signal should be heavily attenuated
        assert np.max(filtered) < 0.1 * np.max(signal)


class TestVibrationMagnitude:
    """Test vibration magnitude calculation."""
    
    def test_magnitude_calculation(self):
        """Test correct magnitude computation."""
        # Simple 3-4-5 triangle
        x = np.array([3.0, 0.0, -3.0])
        y = np.array([4.0, 0.0, -4.0])
        z = np.array([0.0, 5.0, 0.0])
        
        mag = compute_vibration_magnitude(x, y, z)
        
        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_almost_equal(mag, expected)
    
    def test_zero_inputs(self):
        """Test with zero inputs."""
        x = np.zeros(100)
        y = np.zeros(100)
        z = np.zeros(100)
        
        mag = compute_vibration_magnitude(x, y, z)
        
        assert np.all(mag == 0)


class TestGravityRemoval:
    """Test gravity removal functionality."""
    
    def test_constant_gravity(self):
        """Test removal of constant gravity."""
        # Signal with gravity on z-axis
        n_samples = 1000
        x = 0.1 * np.random.randn(n_samples)
        y = 0.1 * np.random.randn(n_samples)
        z = 9.81 + 0.1 * np.random.randn(n_samples)
        
        x_ng, y_ng, z_ng = remove_gravity(x, y, z)
        
        # Mean should be near zero
        assert abs(np.mean(x_ng)) < 0.01
        assert abs(np.mean(y_ng)) < 0.01
        assert abs(np.mean(z_ng)) < 0.01


class TestQualityMetrics:
    """Test quality metrics computation."""
    
    def test_metrics_calculation(self):
        """Test all metrics are computed correctly."""
        # Create test signal
        signal = np.random.randn(1000)
        
        metrics = compute_quality_metrics(signal)
        
        # Check all expected metrics exist
        expected_keys = ['rms', 'peak', 'mean', 'std', 'kurtosis', 
                        'peak_to_rms', 'clipping_ratio', 'is_clipped']
        for key in expected_keys:
            assert key in metrics
        
        # Verify RMS calculation
        expected_rms = np.sqrt(np.mean(signal**2))
        assert abs(metrics['rms'] - expected_rms) < 1e-6
    
    def test_clipping_detection(self):
        """Test clipping detection."""
        # Create signal with clipping
        signal = np.random.randn(1000)
        signal[0:50] = 10.0  # Clip first 50 samples
        
        metrics = compute_quality_metrics(signal)
        
        assert metrics['is_clipped'] == True
        assert metrics['clipping_ratio'] >= 0.05  # At least 5% clipped


class TestDetrending:
    """Test signal detrending."""
    
    def test_linear_detrend(self):
        """Test linear detrending."""
        # Create signal with linear trend
        t = np.linspace(0, 10, 1000)
        trend = 2 * t + 5
        signal = trend + 0.1 * np.sin(2 * np.pi * 5 * t)
        
        detrended = detrend_signal(signal, method='linear')
        
        # Should remove linear trend
        assert abs(np.mean(detrended)) < 0.1
        # Should preserve oscillation
        assert np.std(detrended) > 0.05


class TestSyntheticData:
    """Test with synthetic data to verify SNR requirements."""
    
    def test_25hz_sine_burst(self):
        """Test 25 Hz sine burst achieves required SNR."""
        fs = 200
        duration = 10
        t = np.arange(0, duration, 1/fs)
        
        # 25 Hz sine wave (1500 RPM)
        signal = np.sin(2 * np.pi * 25 * t)
        
        # Add noise for 30 dB SNR
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(30/10))
        noise = np.sqrt(noise_power) * np.random.randn(len(t))
        noisy_signal = signal + noise
        
        # Process through high-pass filter
        filtered = high_pass_filter(noisy_signal, fs, cutoff=5.0)
        
        # Calculate output SNR
        # Find signal component around 25 Hz
        from scipy import signal as sp
        f, psd = sp.welch(filtered, fs, nperseg=1024)
        
        # Find peak around 25 Hz
        idx_25hz = np.argmin(np.abs(f - 25))
        signal_band = slice(idx_25hz - 2, idx_25hz + 3)
        noise_band = slice(0, len(f))
        
        signal_power_out = np.max(psd[signal_band])
        noise_power_out = np.median(psd[noise_band])
        
        snr_out = 10 * np.log10(signal_power_out / noise_power_out)
        
        # Should achieve at least 25 dB SNR
        assert snr_out >= 25, f"SNR {snr_out:.1f} dB < 25 dB requirement"


# Test anti-aliasing filter
def test_anti_alias_filter():
    """Test anti-aliasing filter performance."""
    fs = 200
    t = np.arange(0, 1, 1/fs)
    
    # Create signal with high frequency content
    signal = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 90 * t)
    
    # Apply anti-alias filter
    filtered = apply_anti_alias_filter(signal, fs, cutoff=85.0)
    
    # Use FFT to check frequency content
    from scipy.fft import fft, fftfreq
    N = len(filtered)
    yf = fft(filtered)
    xf = fftfreq(N, 1/fs)[:N//2]
    
    # Check attenuation at 90 Hz
    idx_90hz = np.argmin(np.abs(xf - 90))
    idx_30hz = np.argmin(np.abs(xf - 30))
    
    power_90hz = np.abs(yf[idx_90hz])**2
    power_30hz = np.abs(yf[idx_30hz])**2
    
    # 90 Hz should be attenuated by at least 20 dB relative to 30 Hz
    attenuation_db = 10 * np.log10(power_90hz / power_30hz)
    assert attenuation_db < -20, f"Insufficient attenuation: {attenuation_db:.1f} dB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])