"""
Simple validation script for WP-2 implementation.
Tests key functionality without requiring pytest.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from spectral import (
    welch_psd, find_peaks_in_psd, compute_snr, 
    extract_rpm_from_vibration, rpm_from_frequency
)


def test_clean_sine_wave():
    """Test RPM extraction from clean sine wave."""
    print("\n=== Test 1: Clean Sine Wave ===")
    
    fs = 200  # Hz
    duration = 10  # seconds
    rpm_true = 1500  # RPM
    f_true = rpm_true / 60  # Hz = 25 Hz
    
    # Generate signal
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = 2.0 * np.sin(2 * np.pi * f_true * t)
    
    # Configuration
    config = {
        'fs': fs,
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
    
    # Extract RPM
    rpm_frame = extract_rpm_from_vibration(
        signal, fs, config, timestamp=5.0, sensor_id='test'
    )
    
    if rpm_frame:
        print(f"  True RPM: {rpm_true}")
        print(f"  Estimated RPM: {rpm_frame.rpm:.1f}")
        print(f"  Error: {abs(rpm_frame.rpm - rpm_true):.1f} RPM")
        print(f"  SNR: {rpm_frame.snr_db:.1f} dB")
        print(f"  ✓ PASSED" if abs(rpm_frame.rpm - rpm_true) < 10 else "  ✗ FAILED")
    else:
        print("  ✗ FAILED - No RPM detected")
    
    return rpm_frame is not None and abs(rpm_frame.rpm - rpm_true) < 10


def test_noisy_signal():
    """Test RPM extraction from noisy signal."""
    print("\n=== Test 2: Noisy Signal ===")
    
    fs = 200
    duration = 10
    rpm_true = 1200
    f_true = rpm_true / 60  # 20 Hz
    
    # Generate noisy signal
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * f_true * t) + 0.5 * np.random.randn(len(t))
    
    # Configuration
    config = {
        'fs': fs,
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
    
    # Extract RPM
    rpm_frame = extract_rpm_from_vibration(
        signal, fs, config, timestamp=5.0, sensor_id='test'
    )
    
    if rpm_frame:
        print(f"  True RPM: {rpm_true}")
        print(f"  Estimated RPM: {rpm_frame.rpm:.1f}")
        print(f"  Error: {abs(rpm_frame.rpm - rpm_true):.1f} RPM")
        print(f"  SNR: {rpm_frame.snr_db:.1f} dB")
        print(f"  ✓ PASSED" if abs(rpm_frame.rpm - rpm_true) < 20 else "  ✗ FAILED")
    else:
        print("  ✗ FAILED - No RPM detected")
    
    return rpm_frame is not None and abs(rpm_frame.rpm - rpm_true) < 20


def test_harmonic_signal():
    """Test with signal containing harmonics."""
    print("\n=== Test 3: Multi-Harmonic Signal ===")
    
    fs = 200
    duration = 10
    f_fundamental = 12.0  # Hz (720 RPM)
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Create signal with harmonics
    signal = np.sin(2 * np.pi * f_fundamental * t)  # Fundamental
    signal += 0.7 * np.sin(2 * np.pi * 2 * f_fundamental * t)  # 2nd harmonic
    signal += 0.3 * np.sin(2 * np.pi * 3 * f_fundamental * t)  # 3rd harmonic
    
    # Configuration
    config = {
        'fs': fs,
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
    
    # Extract RPM
    rpm_frame = extract_rpm_from_vibration(
        signal, fs, config, timestamp=5.0, sensor_id='test'
    )
    
    rpm_true = 720
    if rpm_frame:
        print(f"  True RPM: {rpm_true}")
        print(f"  Estimated RPM: {rpm_frame.rpm:.1f}")
        print(f"  Error: {abs(rpm_frame.rpm - rpm_true):.1f} RPM")
        print(f"  SNR: {rpm_frame.snr_db:.1f} dB")
        print(f"  Harmonics detected: {len(rpm_frame.metadata.get('harmonics', {}))}")
        print(f"  ✓ PASSED" if abs(rpm_frame.rpm - rpm_true) < 10 else "  ✗ FAILED")
    else:
        print("  ✗ FAILED - No RPM detected")
    
    return rpm_frame is not None and abs(rpm_frame.rpm - rpm_true) < 10


def test_psd_peak_detection():
    """Test PSD and peak detection."""
    print("\n=== Test 4: PSD Peak Detection ===")
    
    # Create synthetic PSD with known peaks
    freqs = np.linspace(0, 100, 1000)
    psd = np.ones_like(freqs) * 1e-6  # Noise floor
    
    # Add peaks
    peak_freqs = [15.0, 30.0, 45.0]  # Fundamental and harmonics
    peak_amps = [1e-3, 5e-4, 2e-4]
    
    for f, amp in zip(peak_freqs, peak_amps):
        idx = np.argmin(np.abs(freqs - f))
        psd[idx] = amp
    
    # Find peaks
    peaks = find_peaks_in_psd(freqs, psd, min_freq=10, max_freq=50)
    
    print(f"  Found {len(peaks)} peaks")
    for i, peak in enumerate(peaks):
        print(f"  Peak {i+1}: {peak['freq']:.1f} Hz, "
              f"{peak['amplitude_db']:.1f} dB, "
              f"height above noise: {peak['height_above_noise']:.1f} dB")
    
    # Check if all peaks found
    found_freqs = [p['freq'] for p in peaks]
    all_found = all(any(abs(f - pf) < 0.5 for pf in found_freqs) for f in peak_freqs)
    
    print(f"  ✓ PASSED" if all_found else "  ✗ FAILED")
    return all_found


def main():
    """Run all validation tests."""
    print("WP-2 Validation Tests")
    print("=" * 50)
    
    tests = [
        test_clean_sine_wave,
        test_noisy_signal,
        test_harmonic_signal,
        test_psd_peak_detection
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Summary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All tests passed! WP-2 implementation is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()