#!/usr/bin/env python3
"""
Generate test plots for WP-3 STFT implementation.

This script creates visualizations of the key test cases including:
- Basic STFT functionality
- Edge effect handling
- SNR gating behavior
- Triangular ramp test
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Import RPM estimation modules
from src.analysis.rpm.spectral import stft_mag, extract_rpm_stft
from src.analysis.rpm.tracking import smooth_rpm_series
import matplotlib.gridspec as gridspec


def generate_basic_stft_plot(output_dir: Path):
    """Generate plot showing basic STFT functionality."""
    # Create test signal: 25 Hz sine wave with frequency change
    fs = 200
    duration = 8.0
    t = np.arange(0, duration, 1/fs)
    
    # First 4 seconds: 25 Hz (1500 RPM)
    # Last 4 seconds: 40 Hz (2400 RPM)
    freq1, freq2 = 25.0, 40.0
    signal = np.concatenate([
        np.sin(2 * np.pi * freq1 * t[:len(t)//2]),
        np.sin(2 * np.pi * freq2 * t[len(t)//2:])
    ])
    
    # Add some noise
    signal += 0.1 * np.random.randn(len(signal))
    
    # Compute STFT
    times, freqs, magnitude = stft_mag(
        signal, fs, 
        win_sec=1.0, 
        hop_sec=0.25,
        window='hann',
        edge_method='mirror'
    )
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1])
    
    # Plot 1: Original signal
    ax1 = plt.subplot(gs[0])
    ax1.plot(t, signal, 'b-', linewidth=0.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('WP-3 STFT Test: Frequency Step Change (1500 → 2400 RPM)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=4.0, color='r', linestyle='--', alpha=0.5, label='Frequency change')
    ax1.legend()
    
    # Plot 2: STFT spectrogram
    ax2 = plt.subplot(gs[1])
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    im = ax2.pcolormesh(times, freqs, magnitude_db, 
                       shading='gouraud', cmap='viridis',
                       vmin=np.max(magnitude_db) - 40)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_ylim([0, 60])
    ax2.axhline(y=25, color='w', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=40, color='w', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(1, 27, '1500 RPM', color='white', fontsize=10)
    ax2.text(6, 42, '2400 RPM', color='white', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Magnitude (dB)')
    
    # Plot 3: Peak frequency tracking
    ax3 = plt.subplot(gs[2])
    peak_freqs = []
    for i in range(magnitude.shape[1]):
        spectrum = magnitude[:, i]
        # Find peak in expected range (10-50 Hz)
        freq_mask = (freqs >= 10) & (freqs <= 50)
        if np.any(freq_mask):
            peak_idx = np.argmax(spectrum[freq_mask])
            peak_freq = freqs[freq_mask][peak_idx]
            peak_freqs.append(peak_freq * 60)  # Convert to RPM
        else:
            peak_freqs.append(np.nan)
    
    ax3.plot(times, peak_freqs, 'b.-', markersize=4)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('RPM')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([1000, 2800])
    
    plt.tight_layout()
    output_path = output_dir / 'stft_basic_functionality.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def generate_edge_handling_plot(output_dir: Path):
    """Generate plot comparing different edge handling methods."""
    # Short signal to emphasize edge effects
    fs = 200
    signal = np.sin(2 * np.pi * 30 * np.linspace(0, 2, 400))  # 2 seconds
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('WP-3 STFT Edge Handling Comparison', fontsize=14)
    
    edge_methods = ['mirror', 'wrap', 'trim']
    colors = ['blue', 'green', 'red']
    
    for idx, (method, color) in enumerate(zip(edge_methods, colors)):
        ax = axes[idx]
        
        # Compute STFT with different edge handling
        times, freqs, magnitude = stft_mag(
            signal, fs,
            win_sec=1.0,
            hop_sec=0.25,
            edge_method=method
        )
        
        # Plot spectrogram
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        im = ax.pcolormesh(times, freqs, magnitude_db,
                          shading='gouraud', cmap='viridis',
                          vmin=np.max(magnitude_db) - 40)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim([0, 50])
        ax.set_title(f'Edge Method: {method.capitalize()}')
        ax.axhline(y=30, color='white', linestyle='--', alpha=0.5)
        
        # Mark edge regions
        ax.axvspan(0, 0.5, alpha=0.2, color='red', label='Edge region')
        ax.axvspan(times[-1]-0.5, times[-1], alpha=0.2, color='red')
        
        if idx == 0:
            ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    output_path = output_dir / 'stft_edge_handling_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def generate_snr_gating_plot(output_dir: Path):
    """Generate plot showing SNR gating behavior."""
    # Create config
    config = {
        'fs': 200,
        'wp3': {
            'quality': {'min_snr_db': 10.0},
            'stft': {'win_sec': 1.0, 'hop_sec': 0.25}
        },
        'peak_detection': {'noise_floor_db': 3.0},
        'snr': {'band_hz': 3.0, 'exclude_hz': 0.5}
    }
    
    # Create signal with varying SNR
    fs = config['fs']
    duration = 8.0
    t = np.arange(0, duration, 1/fs)
    
    # Base signal: 25 Hz (1500 RPM)
    freq = 25.0
    signal = np.sin(2 * np.pi * freq * t)
    
    # Variable noise level
    noise = np.random.randn(len(t))
    noise_envelope = np.ones_like(t)
    # Create regions of different SNR
    noise_envelope[int(2*fs):int(3*fs)] = 5.0   # Low SNR region
    noise_envelope[int(5*fs):int(6*fs)] = 10.0  # Very low SNR region
    
    noisy_signal = signal + noise * noise_envelope
    
    # Extract RPM with SNR gating
    rpm_series = extract_rpm_stft(
        noisy_signal,
        fs=fs,
        config=config,
        start_time=0.0,
        sensor_id='test'
    )
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('WP-3 Early SNR Gating Demonstration', fontsize=14)
    
    # Plot 1: Signal with noise regions
    ax1 = axes[0]
    ax1.plot(t, noisy_signal, 'b-', linewidth=0.5, alpha=0.7)
    ax1.fill_between(t, -3*noise_envelope, 3*noise_envelope, 
                     alpha=0.2, color='red', label='Noise envelope')
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim([-15, 15])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: STFT spectrogram
    ax2 = axes[1]
    times, freqs, magnitude = stft_mag(
        noisy_signal, fs, 
        win_sec=1.0, hop_sec=0.25
    )
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    im = ax2.pcolormesh(times, freqs, magnitude_db,
                       shading='gouraud', cmap='viridis',
                       vmin=np.max(magnitude_db) - 40)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_ylim([0, 50])
    ax2.axhline(y=25, color='white', linestyle='--', alpha=0.5)
    
    # Plot 3: SNR over time
    ax3 = axes[2]
    times_rpm, rpms, snrs = rpm_series.to_arrays()
    ax3.plot(times_rpm, snrs, 'g.-', markersize=4)
    ax3.axhline(y=10, color='r', linestyle='--', label='SNR threshold')
    ax3.set_ylabel('SNR (dB)')
    ax3.set_ylim([0, 30])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RPM with gating
    ax4 = axes[3]
    valid_mask = ~np.isnan(rpms)
    
    # Plot all points
    ax4.scatter(times_rpm, np.ones_like(times_rpm) * 1500, 
               c='lightgray', s=50, marker='o', label='Time bins')
    
    # Plot valid estimates
    if np.any(valid_mask):
        ax4.scatter(times_rpm[valid_mask], rpms[valid_mask], 
                   c='blue', s=50, marker='o', label='Valid estimates')
    
    # Plot gated estimates
    gated_mask = ~valid_mask
    if np.any(gated_mask):
        ax4.scatter(times_rpm[gated_mask], np.ones(np.sum(gated_mask)) * 1500, 
                   c='red', s=50, marker='x', label='Gated (low SNR)')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('RPM')
    ax4.set_ylim([1400, 1600])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add annotations for SNR regions
    ax1.axvspan(2, 3, alpha=0.1, color='orange', label='Low SNR')
    ax1.axvspan(5, 6, alpha=0.1, color='red', label='Very low SNR')
    
    plt.tight_layout()
    output_path = output_dir / 'stft_snr_gating_behavior.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def generate_triangular_ramp_plot(output_dir: Path):
    """Generate comprehensive triangular ramp test plot."""
    fs = 200
    duration = 10.0
    t = np.arange(0, duration, 1/fs)
    
    # Create triangular RPM profile
    rpm_profile = np.zeros_like(t)
    mid_point = len(t) // 2
    
    # Ramp up: 500 to 2000 RPM in 5 seconds
    rpm_profile[:mid_point] = np.linspace(500, 2000, mid_point)
    # Ramp down: 2000 to 500 RPM in 5 seconds
    rpm_profile[mid_point:] = np.linspace(2000, 500, len(t) - mid_point)
    
    # Convert to frequency
    freq_profile = rpm_profile / 60.0
    
    # Generate chirp signal
    phase = 2 * np.pi * np.cumsum(freq_profile) / fs
    signal = np.sin(phase)
    
    # Add moderate noise
    signal += 0.15 * np.random.randn(len(signal))
    
    # Config for testing
    config = {
        'fs': fs,
        'wp3': {
            'quality': {'min_snr_db': 5.0},
            'stft': {'win_sec': 1.0, 'hop_sec': 0.25},
            'smoothing': {'enabled': True, 'method': 'polynomial', 'window_size': 5}
        },
        'peak_detection': {'noise_floor_db': 3.0},
        'snr': {'band_hz': 3.0, 'exclude_hz': 0.5}
    }
    
    # Extract RPM
    rpm_series = extract_rpm_stft(
        signal,
        fs=fs,
        config=config,
        start_time=0.0,
        sensor_id='test'
    )
    
    # Apply smoothing
    times_rpm, rpms, snrs = rpm_series.to_arrays()
    smoothed_rpm = smooth_rpm_series(
        times_rpm, rpms,
        method='polynomial',
        window=5,
        high_rate_threshold=150.0
    )
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1.5, 1, 1])
    
    # Plot 1: Signal overview
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(t[:1000], signal[:1000], 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('WP-3 Triangular Ramp Test: 500→2000→500 RPM over 10s', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 5])
    
    # Plot 2: STFT spectrogram
    ax2 = plt.subplot(gs[1, :])
    times, freqs, magnitude = stft_mag(signal, fs, win_sec=1.0, hop_sec=0.25)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    im = ax2.pcolormesh(times, freqs, magnitude_db,
                       shading='gouraud', cmap='viridis',
                       vmin=np.max(magnitude_db) - 40)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_ylim([0, 40])
    
    # Overlay true frequency profile
    ax2.plot(t, freq_profile, 'w--', linewidth=2, alpha=0.8, label='True frequency')
    ax2.legend(loc='upper right')
    
    # Plot 3: RPM tracking
    ax3 = plt.subplot(gs[2, :])
    
    # True RPM profile
    ax3.plot(t, rpm_profile, 'k--', linewidth=2, alpha=0.5, label='True RPM')
    
    # STFT estimates
    valid_mask = ~np.isnan(rpms)
    if np.any(valid_mask):
        ax3.scatter(times_rpm[valid_mask], rpms[valid_mask], 
                   c='blue', s=20, alpha=0.6, label='STFT estimates')
    
    # Smoothed estimates
    smoothed_valid = ~np.isnan(smoothed_rpm)
    if np.any(smoothed_valid):
        ax3.plot(times_rpm[smoothed_valid], smoothed_rpm[smoothed_valid], 
                'r-', linewidth=2, label='Smoothed')
    
    ax3.set_ylabel('RPM')
    ax3.set_ylim([300, 2200])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Tracking error
    ax4 = plt.subplot(gs[3, 0])
    if np.any(valid_mask):
        # Interpolate true RPM at measurement points
        true_rpm_at_meas = np.interp(times_rpm[valid_mask], t, rpm_profile)
        error = rpms[valid_mask] - true_rpm_at_meas
        
        ax4.scatter(times_rpm[valid_mask], error, c='blue', s=10, alpha=0.6)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Error (RPM)')
        ax4.set_ylim([-100, 100])
        ax4.grid(True, alpha=0.3)
        
        # Add RMSE text
        rmse = np.sqrt(np.mean(error**2))
        ax4.text(0.05, 0.95, f'RMSE: {rmse:.1f} RPM', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Plot 5: RPM rate of change
    ax5 = plt.subplot(gs[3, 1])
    if len(times_rpm) > 1 and np.sum(valid_mask) > 1:
        # Calculate rate of change
        dt = np.diff(times_rpm[valid_mask])
        drpm = np.diff(rpms[valid_mask])
        rpm_rate = drpm / dt
        
        ax5.plot(times_rpm[valid_mask][1:], rpm_rate, 'g-', linewidth=1)
        ax5.axhline(y=150, color='r', linestyle='--', alpha=0.5, label='High-rate threshold')
        ax5.axhline(y=-150, color='r', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('RPM Rate (RPM/s)')
        ax5.set_ylim([-400, 400])
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'stft_triangular_ramp_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def generate_smoothing_comparison_plot(output_dir: Path):
    """Generate plot comparing different smoothing methods."""
    # Create test data with high-rate change and noise
    time = np.linspace(0, 10, 200)
    
    # Base RPM profile with sudden changes
    rpm_true = np.zeros_like(time)
    rpm_true[0:40] = 1000  # Steady
    rpm_true[40:80] = np.linspace(1000, 2000, 40)  # Ramp up
    rpm_true[80:120] = 2000  # Steady
    rpm_true[120:160] = np.linspace(2000, 1500, 40)  # Ramp down
    rpm_true[160:] = 1500  # Steady
    
    # Add noise
    rpm_noisy = rpm_true + 50 * np.random.randn(len(rpm_true))
    
    # Apply different smoothing methods
    methods = ['polynomial', 'median', 'moving_avg']
    colors = ['red', 'green', 'blue']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('WP-3 Smoothing Methods Comparison', fontsize=14)
    
    # Plot 1: All methods
    ax1 = axes[0]
    ax1.plot(time, rpm_true, 'k--', linewidth=2, alpha=0.5, label='True RPM')
    ax1.scatter(time, rpm_noisy, c='lightgray', s=10, alpha=0.5, label='Noisy measurements')
    
    smoothed_results = {}
    for method, color in zip(methods, colors):
        smoothed = smooth_rpm_series(
            time, rpm_noisy,
            method=method,
            window=7,
            high_rate_threshold=100.0
        )
        smoothed_results[method] = smoothed
        ax1.plot(time, smoothed, color=color, linewidth=2, 
                label=f'{method.capitalize()} smoothing')
    
    ax1.set_ylabel('RPM')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([800, 2200])
    
    # Plot 2: Smoothing intensity (where smoothing was applied)
    ax2 = axes[1]
    
    # Calculate RPM rate
    dt = np.diff(time)
    drpm_true = np.diff(rpm_true)
    rpm_rate = np.abs(drpm_true / dt)
    
    # Show high-rate regions
    ax2.fill_between(time[1:], 0, 300, 
                    where=(rpm_rate > 100),
                    alpha=0.3, color='orange',
                    label='High-rate regions (>100 RPM/s)')
    
    ax2.plot(time[1:], rpm_rate, 'k-', linewidth=1, label='RPM rate')
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Smoothing threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('|RPM Rate| (RPM/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 300])
    
    plt.tight_layout()
    output_path = output_dir / 'stft_smoothing_methods_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def generate_summary_plot(output_dir: Path):
    """Generate a summary plot showcasing all WP-3 features."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create a 3x3 grid for different features
    gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
    
    # Feature 1: Temporal resolution comparison
    ax1 = plt.subplot(gs[0, 0])
    times_welch = np.arange(0, 60, 30)  # 30s windows
    times_stft = np.arange(0, 60, 0.25)  # 0.25s hop
    
    ax1.scatter(times_welch, np.ones_like(times_welch), 
               s=100, c='blue', marker='s', label='WP-2 Welch (30s)')
    ax1.scatter(times_stft[::4], np.ones_like(times_stft[::4]) * 0.5,
               s=20, c='red', marker='o', label='WP-3 STFT (0.25s)')
    ax1.set_xlim([0, 60])
    ax1.set_ylim([0, 1.5])
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Temporal Resolution')
    ax1.legend()
    ax1.set_yticks([])
    
    # Feature 2: Frequency resolution
    ax2 = plt.subplot(gs[0, 1])
    freqs_welch = np.arange(0, 50, 0.167)  # 0.167 Hz resolution
    freqs_stft = np.arange(0, 50, 1.0)     # 1 Hz resolution
    
    ax2.vlines(freqs_welch, 0, 1, colors='blue', alpha=0.3, linewidth=1, label='Welch (0.167 Hz)')
    ax2.vlines(freqs_stft, 0, 0.8, colors='red', alpha=0.5, linewidth=2, label='STFT (1 Hz)')
    ax2.set_xlim([20, 30])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_title('Frequency Resolution')
    ax2.legend()
    ax2.set_yticks([])
    
    # Feature 3: SNR gating
    ax3 = plt.subplot(gs[0, 2])
    snr_values = np.linspace(0, 20, 100)
    gating = snr_values >= 10
    
    ax3.fill_between(snr_values[~gating], 0, 1, color='red', alpha=0.3, label='Gated')
    ax3.fill_between(snr_values[gating], 0, 1, color='green', alpha=0.3, label='Valid')
    ax3.axvline(x=10, color='k', linestyle='--', label='Threshold')
    ax3.set_xlabel('SNR (dB)')
    ax3.set_title('Early SNR Gating')
    ax3.legend()
    ax3.set_ylim([0, 1])
    ax3.set_yticks([])
    
    # Feature 4: Edge handling methods
    ax4 = plt.subplot(gs[1, :])
    t = np.linspace(0, 2, 200)
    signal = np.sin(2 * np.pi * 5 * t)
    
    # Show different padding
    pad_len = 50
    mirror_pad = np.concatenate([signal[:pad_len][::-1], signal, signal[-pad_len:][::-1]])
    wrap_pad = np.concatenate([signal[-pad_len:], signal, signal[:pad_len]])
    
    t_pad = np.linspace(-0.5, 2.5, len(mirror_pad))
    
    ax4.plot(t, signal, 'k-', linewidth=2, label='Original')
    ax4.plot(t_pad, mirror_pad, 'b--', alpha=0.7, label='Mirror padding')
    ax4.plot(t_pad, wrap_pad, 'r--', alpha=0.7, label='Wrap padding')
    ax4.axvspan(-0.5, 0, alpha=0.2, color='gray')
    ax4.axvspan(2, 2.5, alpha=0.2, color='gray')
    ax4.set_xlim([-0.5, 2.5])
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Edge Handling Methods')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Feature 5: Processing pipeline
    ax5 = plt.subplot(gs[2, :])
    ax5.text(0.5, 0.9, 'WP-3 Processing Pipeline', ha='center', fontsize=14, weight='bold')
    
    pipeline_steps = [
        '1. Load WP-1 data',
        '2. Verify anti-aliasing',
        '3. Apply STFT (1s window, 0.25s hop)',
        '4. Extract RPM per time slice',
        '5. Calculate SNR & gate low-confidence',
        '6. Apply smoothing to high-rate regions',
        '7. Save HDF5 with metadata'
    ]
    
    for i, step in enumerate(pipeline_steps):
        ax5.text(0.1, 0.8 - i*0.1, step, fontsize=11)
    
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    ax5.axis('off')
    
    # Add overall title
    fig.suptitle('WP-3 STFT Implementation - Feature Summary', fontsize=16, y=0.98)
    
    plt.tight_layout()
    output_path = output_dir / 'wp3_feature_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def main():
    """Generate all WP-3 test plots."""
    # Create output directory
    output_dir = Path(__file__).parent / 'results' / 'wp3' / 'test_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating WP-3 test plots...")
    
    # Generate all plots
    generate_basic_stft_plot(output_dir)
    generate_edge_handling_plot(output_dir)
    generate_snr_gating_plot(output_dir)
    generate_triangular_ramp_plot(output_dir)
    generate_smoothing_comparison_plot(output_dir)
    generate_summary_plot(output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    
    # Create an index HTML file for easy viewing
    html_content = """
    <html>
    <head>
        <title>WP-3 STFT Test Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .plot { margin: 20px 0; border: 1px solid #ddd; padding: 10px; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>WP-3 STFT Implementation Test Results</h1>
        
        <div class="plot">
            <h2>1. Basic STFT Functionality</h2>
            <p>Demonstrates STFT tracking of frequency step change (1500 → 2400 RPM)</p>
            <img src="stft_basic_functionality.png">
        </div>
        
        <div class="plot">
            <h2>2. Edge Handling Comparison</h2>
            <p>Shows different edge handling methods (mirror, wrap, trim)</p>
            <img src="stft_edge_handling_comparison.png">
        </div>
        
        <div class="plot">
            <h2>3. SNR Gating Behavior</h2>
            <p>Demonstrates early SNR gating removing low-confidence estimates</p>
            <img src="stft_snr_gating_behavior.png">
        </div>
        
        <div class="plot">
            <h2>4. Triangular Ramp Test</h2>
            <p>Key validation: 500→2000→500 RPM tracking over 10 seconds</p>
            <img src="stft_triangular_ramp_test.png">
        </div>
        
        <div class="plot">
            <h2>5. Smoothing Methods Comparison</h2>
            <p>Comparison of polynomial, median, and moving average smoothing</p>
            <img src="stft_smoothing_methods_comparison.png">
        </div>
        
        <div class="plot">
            <h2>6. Feature Summary</h2>
            <p>Overview of all WP-3 features and processing pipeline</p>
            <img src="wp3_feature_summary.png">
        </div>
    </body>
    </html>
    """
    
    index_path = output_dir / 'index.html'
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created index: {index_path}")


if __name__ == '__main__':
    main()