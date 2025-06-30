"""
Unit tests for spectral analysis module (WP-2).

Tests cover:
- Welch PSD computation
- Peak detection
- SNR calculation
- Harmonic extraction
- RPM extraction from synthetic signals
"""

import pytest
import numpy as np
from scipy import signal

from src.analysis.rpm.spectral import (
    welch_psd, find_peaks_in_psd, compute_snr, extract_harmonics,
    rpm_from_frequency, frequency_from_rpm, identify_fundamental,
    extract_rpm_from_vibration
)
from src.analysis.rpm.tracking import RPMFrame


class TestWelchPSD:
    """Test Welch PSD functionality."""
    
    def test_basic_welch(self):
        """Test basic Welch PSD computation."""
        # Create a simple sine wave
        fs = 200  # Hz
        duration = 10  # seconds
        f_signal = 25  # Hz (1500 RPM)
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = np.sin(2 * np.pi * f_signal * t)
        
        # Compute PSD
        freqs, psd = welch_psd(signal, fs=fs, win_sec=6.0, overlap=0.5)
        
        # Check frequency resolution
        freq_res = freqs[1] - freqs[0]
        assert abs(freq_res - fs / (6.0 * fs)) < 0.01
        
        # Check that we limit to 100 Hz
        assert freqs[-1] <= 100.0
        
        # Check peak at signal frequency
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - f_signal) < 1.0  # Within 1 Hz
    
    def test_window_parameters(self):
        """Test different window parameters."""
        fs = 200
        data = np.random.randn(2000)  # 10 seconds of noise
        
        # Test different window lengths
        freqs1, psd1 = welch_psd(data, fs, win_sec=2.0)
        freqs2, psd2 = welch_psd(data, fs, win_sec=4.0)
        
        # Longer window should have better frequency resolution
        assert len(freqs2) > len(freqs1)
        assert (freqs1[1] - freqs1[0]) > (freqs2[1] - freqs2[0])


class TestPeakDetection:
    """Test peak detection in PSD."""
    
    def test_single_peak(self):
        """Test detection of a single peak."""
        # Create synthetic PSD with one peak
        freqs = np.linspace(0, 100, 1000)
        psd = np.ones_like(freqs) * 1e-6  # Noise floor
        
        # Add peak at 30 Hz
        peak_idx = np.argmin(np.abs(freqs - 30))
        psd[peak_idx] = 1e-3  # 30 dB above noise
        
        # Find peaks
        peaks = find_peaks_in_psd(freqs, psd, min_freq=10, max_freq=50)
        
        assert len(peaks) == 1
        assert abs(peaks[0]['freq'] - 30.0) < 0.2
        assert peaks[0]['height_above_noise'] > 25  # Should be ~30 dB
    
    def test_multiple_peaks(self):
        """Test detection of multiple peaks."""
        # Create synthetic PSD with harmonics
        freqs = np.linspace(0, 100, 1000)
        psd = np.ones_like(freqs) * 1e-6
        
        # Add fundamental and harmonics
        for f, amp in [(15, 1e-3), (30, 5e-4), (45, 2e-4)]:
            idx = np.argmin(np.abs(freqs - f))
            psd[idx] = amp
        
        peaks = find_peaks_in_psd(freqs, psd, min_freq=10, max_freq=50)
        
        assert len(peaks) == 3
        # Should be sorted by amplitude
        assert peaks[0]['freq'] < 20  # Fundamental is strongest
        
    def test_noise_floor_filtering(self):
        """Test that peaks below noise floor are filtered."""
        freqs = np.linspace(0, 100, 1000)
        psd = np.ones_like(freqs) * 1e-6
        
        # Add weak peak barely above noise
        idx = np.argmin(np.abs(freqs - 25))
        psd[idx] = 1.5e-6  # Only 1.8 dB above noise
        
        peaks = find_peaks_in_psd(freqs, psd, min_freq=10, max_freq=50, 
                                 noise_floor_db=3.0)
        
        assert len(peaks) == 0  # Should not detect weak peak


class TestSNRCalculation:
    """Test SNR computation."""
    
    def test_snr_clean_signal(self):
        """Test SNR of clean signal."""
        freqs = np.linspace(0, 100, 1000)
        psd = np.ones_like(freqs) * 1e-6  # Noise floor
        
        # Add strong peak
        peak_freq = 25.0
        peak_idx = np.argmin(np.abs(freqs - peak_freq))
        psd[peak_idx] = 1e-3  # 30 dB above noise
        
        snr = compute_snr(freqs, psd, peak_freq, band_hz=3.0, exclude_hz=0.5)
        
        assert 25 < snr < 35  # Should be around 30 dB
        
    def test_snr_noisy_signal(self):
        """Test SNR with higher noise floor."""
        freqs = np.linspace(0, 100, 1000)
        psd = np.ones_like(freqs) * 1e-4  # Higher noise
        
        peak_freq = 25.0
        peak_idx = np.argmin(np.abs(freqs - peak_freq))
        psd[peak_idx] = 1e-3  # 10 dB above noise
        
        snr = compute_snr(freqs, psd, peak_freq)
        
        assert 8 < snr < 12  # Should be around 10 dB


class TestHarmonicExtraction:
    """Test harmonic analysis."""
    
    def test_harmonic_extraction(self):
        """Test extraction of harmonics."""
        freqs = np.linspace(0, 100, 1000)
        psd = np.ones_like(freqs) * 1e-6
        
        # Add fundamental and harmonics
        fundamental = 12.0  # Hz
        for n in range(1, 6):
            idx = np.argmin(np.abs(freqs - n * fundamental))
            psd[idx] = 1e-3 / n  # Decreasing amplitude
        
        harmonics = extract_harmonics(freqs, psd, fundamental, n_harmonics=5)
        
        assert len(harmonics) == 5
        for n in range(1, 6):
            assert n in harmonics
            assert harmonics[n] > 1e-6  # Above noise floor


class TestFundamentalIdentification:
    """Test fundamental frequency identification."""
    
    def test_simple_fundamental(self):
        """Test with clear fundamental."""
        peaks = [
            {'freq': 15.0, 'amplitude_db': 30},
            {'freq': 30.0, 'amplitude_db': 25},
            {'freq': 45.0, 'amplitude_db': 20}
        ]
        
        fundamental = identify_fundamental(peaks, harmonics_check=True)
        
        assert fundamental['freq'] == 15.0  # Should identify fundamental
        
    def test_suppressed_fundamental(self):
        """Test when fundamental is weaker than 2nd harmonic."""
        peaks = [
            {'freq': 30.0, 'amplitude_db': 35},  # 2nd harmonic strongest
            {'freq': 15.0, 'amplitude_db': 25},  # Fundamental weaker
            {'freq': 45.0, 'amplitude_db': 20}   # 3rd harmonic
        ]
        
        fundamental = identify_fundamental(peaks, harmonics_check=True)
        
        # Should still identify 15 Hz as fundamental due to harmonic scoring
        assert fundamental['freq'] == 15.0


class TestRPMExtraction:
    """Test complete RPM extraction pipeline."""
    
    @pytest.fixture
    def config(self):
        """Standard configuration for testing."""
        return {
            'welch': {
                'win_sec': 6.0,
                'overlap': 0.5
            },
            'peak_detection': {
                'noise_floor_db': 3.0,
                'max_harmonics': 5
            },
            'snr': {
                'band_hz': 3.0,
                'exclude_hz': 0.5
            }
        }
    
    def test_clean_sine_wave(self, config):
        """Test RPM extraction from clean sine wave."""
        fs = 200
        duration = 10
        rpm_true = 1500  # RPM
        f_true = rpm_true / 60  # Hz
        
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = 2.0 * np.sin(2 * np.pi * f_true * t)
        
        rpm_frame = extract_rpm_from_vibration(
            signal, fs, config, timestamp=5.0, sensor_id='test'
        )
        
        assert rpm_frame is not None
        assert abs(rpm_frame.rpm - rpm_true) < 10  # Within 10 RPM
        assert rpm_frame.snr_db > 25  # High SNR for clean signal
        assert rpm_frame.method == 'welch'
        
    def test_noisy_signal(self, config):
        """Test RPM extraction from noisy signal."""
        fs = 200
        duration = 10
        rpm_true = 1200
        f_true = rpm_true / 60
        
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = np.sin(2 * np.pi * f_true * t) + 0.5 * np.random.randn(len(t))
        
        rpm_frame = extract_rpm_from_vibration(
            signal, fs, config, timestamp=5.0, sensor_id='test'
        )
        
        assert rpm_frame is not None
        assert abs(rpm_frame.rpm - rpm_true) < 20  # Within 20 RPM
        assert rpm_frame.snr_db > 5  # Lower SNR due to noise
        
    def test_multi_harmonic_signal(self, config):
        """Test with signal containing harmonics."""
        fs = 200
        duration = 10
        f_fundamental = 12.0  # Hz (720 RPM)
        
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Create signal with harmonics
        signal = np.sin(2 * np.pi * f_fundamental * t)  # Fundamental
        signal += 0.7 * np.sin(2 * np.pi * 2 * f_fundamental * t)  # 2nd harmonic
        signal += 0.3 * np.sin(2 * np.pi * 3 * f_fundamental * t)  # 3rd harmonic
        
        rpm_frame = extract_rpm_from_vibration(
            signal, fs, config, timestamp=5.0, sensor_id='test'
        )
        
        assert rpm_frame is not None
        assert abs(rpm_frame.rpm - 720) < 10  # Should find fundamental
        assert len(rpm_frame.metadata['harmonics']) >= 3  # Should detect harmonics
        
    def test_insufficient_data(self, config):
        """Test with insufficient data length."""
        fs = 200
        signal = np.random.randn(500)  # Only 2.5 seconds (need 6)
        
        rpm_frame = extract_rpm_from_vibration(
            signal, fs, config, timestamp=1.0, sensor_id='test'
        )
        
        assert rpm_frame is None  # Should return None
        
    def test_no_peaks(self, config):
        """Test with pure noise (no peaks)."""
        fs = 200
        duration = 10
        signal = np.random.randn(int(fs * duration)) * 0.1
        
        rpm_frame = extract_rpm_from_vibration(
            signal, fs, config, timestamp=5.0, sensor_id='test'
        )
        
        assert rpm_frame is None  # Should return None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_rpm_frequency_conversion(self):
        """Test RPM to frequency conversion."""
        assert abs(frequency_from_rpm(1800) - 30.0) < 0.001
        assert abs(rpm_from_frequency(25.0) - 1500) < 0.001
        
        # Test round trip
        rpm = 1234.5
        assert abs(rpm_from_frequency(frequency_from_rpm(rpm)) - rpm) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])