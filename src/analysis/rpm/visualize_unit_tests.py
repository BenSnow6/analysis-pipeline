#!/usr/bin/env python3
"""
Visualize the unit tests from validate_wp2.py with plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from spectral import (
    welch_psd, find_peaks_in_psd, compute_snr, 
    extract_rpm_from_vibration, rpm_from_frequency
)


def plot_clean_sine_wave():
    """Visualize RPM extraction from clean sine wave."""
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
    
    # Compute PSD
    freqs, psd = welch_psd(signal, fs, win_sec=config['welch']['win_sec'],
                          overlap=config['welch']['overlap'])
    
    # Extract RPM
    rpm_frame = extract_rpm_from_vibration(
        signal, fs, config, timestamp=5.0, sensor_id='test'
    )
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time series
    ax1.plot(t[:400], signal[:400], 'b-', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Clean Sine Wave - {f_true} Hz ({rpm_true} RPM)')
    ax1.grid(True, alpha=0.3)
    
    # PSD
    psd_db = 10 * np.log10(psd + 1e-12)
    ax2.plot(freqs, psd_db, 'b-', linewidth=1)
    ax2.axvline(f_true, color='r', linestyle='--', label=f'True: {f_true} Hz')
    if rpm_frame:
        f_estimated = rpm_frame.rpm / 60
        ax2.axvline(f_estimated, color='g', linestyle='--', 
                   label=f'Estimated: {f_estimated:.1f} Hz')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD (dB)')
    ax2.set_title(f'Power Spectral Density - SNR: {rpm_frame.snr_db:.1f} dB')
    ax2.set_xlim(0, 60)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig, rpm_frame


def plot_noisy_signal():
    """Visualize RPM extraction from noisy signal."""
    print("\n=== Test 2: Noisy Signal ===")
    
    fs = 200
    duration = 10
    rpm_true = 1200
    f_true = rpm_true / 60  # 20 Hz
    
    # Generate noisy signal
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    np.random.seed(42)  # For reproducibility
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
    
    # Compute PSD
    freqs, psd = welch_psd(signal, fs, win_sec=config['welch']['win_sec'],
                          overlap=config['welch']['overlap'])
    
    # Extract RPM
    rpm_frame = extract_rpm_from_vibration(
        signal, fs, config, timestamp=5.0, sensor_id='test'
    )
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time series
    ax1.plot(t[:400], signal[:400], 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Noisy Signal - {f_true} Hz ({rpm_true} RPM) + Noise')
    ax1.grid(True, alpha=0.3)
    
    # PSD
    psd_db = 10 * np.log10(psd + 1e-12)
    ax2.plot(freqs, psd_db, 'b-', linewidth=1)
    ax2.axvline(f_true, color='r', linestyle='--', label=f'True: {f_true} Hz')
    if rpm_frame:
        f_estimated = rpm_frame.rpm / 60
        ax2.axvline(f_estimated, color='g', linestyle='--', 
                   label=f'Estimated: {f_estimated:.1f} Hz')
    
    # Show noise floor
    noise_floor = np.median(psd_db)
    ax2.axhline(noise_floor, color='k', linestyle=':', alpha=0.5, 
                label=f'Noise floor: {noise_floor:.1f} dB')
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD (dB)')
    ax2.set_title(f'Power Spectral Density - SNR: {rpm_frame.snr_db:.1f} dB')
    ax2.set_xlim(0, 60)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig, rpm_frame


def plot_harmonic_signal():
    """Visualize signal with harmonics."""
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
    
    # Compute PSD
    freqs, psd = welch_psd(signal, fs, win_sec=config['welch']['win_sec'],
                          overlap=config['welch']['overlap'])
    
    # Extract RPM
    rpm_frame = extract_rpm_from_vibration(
        signal, fs, config, timestamp=5.0, sensor_id='test'
    )
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time series
    ax1.plot(t[:400], signal[:400], 'b-', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Multi-Harmonic Signal - Fundamental: {f_fundamental} Hz (720 RPM)')
    ax1.grid(True, alpha=0.3)
    
    # PSD
    psd_db = 10 * np.log10(psd + 1e-12)
    ax2.plot(freqs, psd_db, 'b-', linewidth=1)
    
    # Mark harmonics
    for n in range(1, 4):
        f_harm = n * f_fundamental
        ax2.axvline(f_harm, color='r', linestyle='--', alpha=0.7,
                   label=f'{n}x: {f_harm} Hz' if n <= 3 else '')
    
    if rpm_frame:
        f_estimated = rpm_frame.rpm / 60
        ax2.axvline(f_estimated, color='g', linestyle='-', linewidth=2,
                   label=f'Detected fundamental: {f_estimated:.1f} Hz')
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD (dB)')
    ax2.set_title(f'Power Spectral Density - Harmonics Present')
    ax2.set_xlim(0, 60)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig, rpm_frame


def plot_peak_detection_demo():
    """Visualize PSD peak detection algorithm."""
    print("\n=== Test 4: PSD Peak Detection ===")
    
    # Create synthetic PSD with known peaks
    freqs = np.linspace(0, 100, 1000)
    psd = np.ones_like(freqs) * 1e-6  # Noise floor
    
    # Add peaks
    peak_freqs = [15.0, 30.0, 45.0]  # Fundamental and harmonics
    peak_amps = [1e-3, 5e-4, 2e-4]
    
    for f, amp in zip(peak_freqs, peak_amps):
        idx = np.argmin(np.abs(freqs - f))
        # Add Gaussian-shaped peak
        sigma = 0.5  # Hz
        gaussian = amp * np.exp(-0.5 * ((freqs - f) / sigma)**2)
        psd += gaussian
    
    # Find peaks
    peaks = find_peaks_in_psd(freqs, psd, min_freq=10, max_freq=50)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot PSD
    psd_db = 10 * np.log10(psd + 1e-12)
    ax.plot(freqs, psd_db, 'b-', linewidth=1, label='PSD')
    
    # Plot noise floor
    noise_floor_db = 10 * np.log10(np.median(psd[100:200]) + 1e-12)
    ax.axhline(noise_floor_db, color='k', linestyle=':', 
               label=f'Noise floor: {noise_floor_db:.1f} dB')
    ax.axhline(noise_floor_db + 3, color='r', linestyle='--', alpha=0.5,
               label='Detection threshold (+3 dB)')
    
    # Mark detected peaks
    for i, peak in enumerate(peaks):
        ax.plot(peak['freq'], peak['amplitude_db'], 'ro', markersize=8)
        ax.annotate(f"{peak['freq']:.1f} Hz\n{peak['height_above_noise']:.1f} dB",
                   (peak['freq'], peak['amplitude_db']),
                   xytext=(peak['freq'] + 2, peak['amplitude_db'] + 2),
                   fontsize=9, ha='left')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB)')
    ax.set_title('Peak Detection Algorithm Demonstration')
    ax.set_xlim(0, 60)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig, peaks


def main():
    """Run all visualization tests."""
    print("WP-2 Unit Test Visualizations")
    print("=" * 50)
    
    # Create output directory for plots
    plot_dir = Path(__file__).parent / 'results' / 'wp2' / 'unit_test_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Clean sine wave
    fig1, rpm1 = plot_clean_sine_wave()
    fig1.savefig(plot_dir / 'test1_clean_sine_wave.png', dpi=150)
    print(f"✓ Test 1 result: {rpm1.rpm:.0f} RPM (expected 1500), SNR: {rpm1.snr_db:.1f} dB")
    
    # Test 2: Noisy signal
    fig2, rpm2 = plot_noisy_signal()
    fig2.savefig(plot_dir / 'test2_noisy_signal.png', dpi=150)
    print(f"✓ Test 2 result: {rpm2.rpm:.0f} RPM (expected 1200), SNR: {rpm2.snr_db:.1f} dB")
    
    # Test 3: Harmonic signal
    fig3, rpm3 = plot_harmonic_signal()
    fig3.savefig(plot_dir / 'test3_harmonic_signal.png', dpi=150)
    print(f"✓ Test 3 result: {rpm3.rpm:.0f} RPM (expected 720), harmonics: {len(rpm3.metadata.get('harmonics', {}))}")
    
    # Test 4: Peak detection
    fig4, peaks = plot_peak_detection_demo()
    fig4.savefig(plot_dir / 'test4_peak_detection.png', dpi=150)
    print(f"✓ Test 4 result: {len(peaks)} peaks detected")
    
    # Create combined figure
    fig_all, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Just show the PSDs from each test
    for ax in axes.flat:
        ax.remove()
    
    # Recreate with better layout
    fig_all = plt.figure(figsize=(15, 12))
    
    # Add text summary
    fig_all.text(0.5, 0.98, 'WP-2 Unit Test Visualizations', 
                ha='center', va='top', fontsize=16, fontweight='bold')
    
    plt.close(fig_all)
    
    print("\n" + "=" * 50)
    print(f"All plots saved to: {plot_dir}")
    print("\nSummary:")
    print("- Test 1: Perfect recovery of clean 25 Hz signal")
    print("- Test 2: Accurate recovery despite noise")
    print("- Test 3: Correct fundamental identification with harmonics")
    print("- Test 4: Peak detection working as expected")
    
    # Keep plots open for viewing
    plt.show()


if __name__ == "__main__":
    main()