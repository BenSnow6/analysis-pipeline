#!/usr/bin/env python3
"""
Debug script for testing RPM interpolation functionality.

This script provides tools for debugging and visualizing the interpolation
of missing RPM frames in the multi-sensor fusion pipeline (WP-4).

Usage:
    python debug_interpolation.py --experiment 026_Engine_rpm_sweep --session afternoon
    python debug_interpolation.py --test-synthetic
    python debug_interpolation.py --experiment 026_Engine_rpm_sweep --session afternoon --plot
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import pandas as pd

# Import RPM estimation modules
from src.analysis.rpm.tracking import RPMFrame, RPMTimeSeries
from src.analysis.rpm.fusion import interpolate_missing_frames
from src.analysis.rpm.io import load_config
from src.core.paths import PROCESSED_DATA_DIR, get_experiment_path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_rpm_series(duration: float = 60.0, 
                                fs: float = 1.0,
                                gaps: List[Tuple[float, float]] = None) -> RPMTimeSeries:
    """
    Create synthetic RPM data with specified gaps for testing.
    
    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        gaps: List of (start, end) tuples for gaps in seconds
        
    Returns:
        Synthetic RPMTimeSeries with gaps
    """
    # Create time vector
    times = np.arange(0, duration, 1/fs)
    
    # Create synthetic RPM profile (ramp up, steady, ramp down)
    rpms = np.zeros_like(times)
    t1, t2, t3 = duration * 0.2, duration * 0.7, duration * 0.9
    
    # Ramp up
    mask1 = times <= t1
    rpms[mask1] = 1000 + 1000 * times[mask1] / t1
    
    # Steady state with slight variation
    mask2 = (times > t1) & (times <= t2)
    rpms[mask2] = 2000 + 50 * np.sin(2 * np.pi * 0.1 * times[mask2])
    
    # Ramp down
    mask3 = (times > t2) & (times <= t3)
    rpms[mask3] = 2000 - 1000 * (times[mask3] - t2) / (t3 - t2)
    
    # Low idle
    mask4 = times > t3
    rpms[mask4] = 1000
    
    # Create frames
    frames = []
    for i, t in enumerate(times):
        # Check if in gap
        in_gap = False
        if gaps:
            for gap_start, gap_end in gaps:
                if gap_start <= t <= gap_end:
                    in_gap = True
                    break
        
        if in_gap:
            # Create invalid frame
            frame = RPMFrame(
                time=t,
                rpm=0.0,
                snr_db=0.0,  # Below threshold
                sensor_id="synthetic",
                method="welch",
                confidence=0.0
            )
        else:
            # Create valid frame with some noise
            rpm_noise = np.random.normal(0, 10)
            frame = RPMFrame(
                time=t,
                rpm=rpms[i] + rpm_noise,
                snr_db=25.0 + np.random.normal(0, 2),
                sensor_id="synthetic",
                method="welch",
                confidence=0.9
            )
        
        frames.append(frame)
    
    return RPMTimeSeries(
        frames=frames,
        experiment="synthetic_test",
        session="test",
        sensor_id="synthetic"
    )


def visualize_interpolation(original: RPMTimeSeries, 
                           interpolated: RPMTimeSeries,
                           title: str = "RPM Interpolation Debug"):
    """
    Visualize original vs interpolated RPM data.
    
    Args:
        original: Original time series with gaps
        interpolated: Interpolated time series
        title: Plot title
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Get data arrays
    times_orig, rpms_orig, snrs_orig = original.to_arrays()
    times_interp, rpms_interp, snrs_interp = interpolated.to_arrays()
    
    # Masks for valid data
    valid_orig = np.array([f.is_valid() for f in original.frames])
    valid_interp = np.array([f.is_valid() for f in interpolated.frames])
    interpolated_mask = np.array([f.method == 'interpolated' for f in interpolated.frames])
    
    # Plot 1: RPM values
    ax1.scatter(times_orig[valid_orig], rpms_orig[valid_orig], 
                alpha=0.6, label='Original valid', s=20)
    ax1.scatter(times_orig[~valid_orig], rpms_orig[~valid_orig], 
                alpha=0.3, color='red', label='Original invalid', s=10)
    ax1.plot(times_interp[interpolated_mask], rpms_interp[interpolated_mask], 
             'g-', label='Interpolated', linewidth=2)
    ax1.set_ylabel('RPM')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SNR values
    ax2.scatter(times_orig[valid_orig], snrs_orig[valid_orig], 
                alpha=0.6, label='Original SNR', s=20)
    ax2.axhline(y=10.0, color='r', linestyle='--', label='Validity threshold')
    ax2.scatter(times_interp[interpolated_mask], snrs_interp[interpolated_mask], 
                color='green', alpha=0.6, label='Interpolated SNR', s=20)
    ax2.set_ylabel('SNR (dB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Availability
    window_size = int(5.0 / np.median(np.diff(times_orig)))  # 5-second window
    availability_orig = []
    availability_interp = []
    
    for i in range(len(times_orig)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(times_orig), i + window_size // 2)
        
        avail_orig = np.mean(valid_orig[start_idx:end_idx]) * 100
        avail_interp = np.mean(valid_interp[start_idx:end_idx]) * 100
        
        availability_orig.append(avail_orig)
        availability_interp.append(avail_interp)
    
    ax3.plot(times_orig, availability_orig, label='Original', alpha=0.7)
    ax3.plot(times_interp, availability_interp, label='Interpolated', alpha=0.7)
    ax3.fill_between(times_orig, 0, availability_orig, alpha=0.3)
    ax3.fill_between(times_interp, availability_orig, availability_interp, 
                     alpha=0.3, color='green')
    ax3.set_ylabel('Availability (%)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    
    plt.tight_layout()
    return fig


def analyze_interpolation_quality(original: RPMTimeSeries, 
                                 interpolated: RPMTimeSeries) -> dict:
    """
    Analyze the quality of interpolation.
    
    Args:
        original: Original time series
        interpolated: Interpolated time series
        
    Returns:
        Dictionary with quality metrics
    """
    # Get arrays
    times_orig, rpms_orig, _ = original.to_arrays()
    times_interp, rpms_interp, _ = interpolated.to_arrays()
    
    # Masks
    valid_orig = np.array([f.is_valid() for f in original.frames])
    valid_interp = np.array([f.is_valid() for f in interpolated.frames])
    interpolated_mask = np.array([f.method == 'interpolated' for f in interpolated.frames])
    
    # Calculate metrics
    metrics = {
        'original_availability': original.availability,
        'interpolated_availability': interpolated.availability,
        'availability_gain': interpolated.availability - original.availability,
        'num_gaps_original': count_gaps(valid_orig),
        'num_gaps_interpolated': count_gaps(valid_interp),
        'num_interpolated_frames': np.sum(interpolated_mask),
        'interpolation_fraction': np.sum(interpolated_mask) / len(interpolated_mask) * 100,
    }
    
    # Gap statistics
    gap_lengths = get_gap_lengths(valid_orig, times_orig)
    if gap_lengths:
        metrics['max_gap_length'] = np.max(gap_lengths)
        metrics['mean_gap_length'] = np.mean(gap_lengths)
        metrics['num_gaps_filled'] = metrics['num_gaps_original'] - metrics['num_gaps_interpolated']
    
    # Interpolation smoothness (for synthetic data where we know ground truth)
    if hasattr(original, 'ground_truth'):
        # Calculate interpolation error
        interp_errors = []
        for i, frame in enumerate(interpolated.frames):
            if frame.method == 'interpolated':
                true_rpm = original.ground_truth[i]
                error = abs(frame.rpm - true_rpm)
                interp_errors.append(error)
        
        if interp_errors:
            metrics['mean_interpolation_error'] = np.mean(interp_errors)
            metrics['max_interpolation_error'] = np.max(interp_errors)
    
    return metrics


def count_gaps(valid_mask: np.ndarray) -> int:
    """Count number of gaps in valid data."""
    if len(valid_mask) == 0:
        return 0
    
    # Find transitions from valid to invalid
    transitions = np.diff(np.concatenate([[True], valid_mask, [True]]))
    gap_starts = np.where(transitions == -1)[0]
    
    return len(gap_starts)


def get_gap_lengths(valid_mask: np.ndarray, times: np.ndarray) -> List[float]:
    """Get lengths of all gaps in seconds."""
    if len(valid_mask) == 0:
        return []
    
    gap_lengths = []
    in_gap = False
    gap_start = None
    
    for i, valid in enumerate(valid_mask):
        if not valid and not in_gap:
            # Gap started
            in_gap = True
            gap_start = i
        elif valid and in_gap:
            # Gap ended
            in_gap = False
            if gap_start is not None:
                gap_length = times[i-1] - times[gap_start]
                gap_lengths.append(gap_length)
    
    # Handle gap at end
    if in_gap and gap_start is not None:
        gap_length = times[-1] - times[gap_start]
        gap_lengths.append(gap_length)
    
    return gap_lengths


def test_synthetic():
    """Run synthetic data test."""
    logger.info("Running synthetic interpolation test...")
    
    # Create synthetic data with various gap sizes
    gaps = [
        (10, 12),    # 2-second gap
        (25, 30),    # 5-second gap  
        (40, 48),    # 8-second gap (should not be interpolated with 5s limit)
        (55, 56),    # 1-second gap
    ]
    
    series = create_synthetic_rpm_series(duration=60.0, fs=1.0, gaps=gaps)
    logger.info(f"Created synthetic series: {series.availability:.1f}% availability")
    
    # Test different max_gap settings
    for max_gap in [3.0, 5.0, 10.0]:
        logger.info(f"\nTesting with max_gap={max_gap}s")
        
        # Interpolate
        interpolated = interpolate_missing_frames(series, max_gap_s=max_gap)
        
        # Analyze
        metrics = analyze_interpolation_quality(series, interpolated)
        
        logger.info(f"  Original availability: {metrics['original_availability']:.1f}%")
        logger.info(f"  Interpolated availability: {metrics['interpolated_availability']:.1f}%")
        logger.info(f"  Availability gain: {metrics['availability_gain']:.1f}%")
        logger.info(f"  Gaps filled: {metrics.get('num_gaps_filled', 'N/A')}")
        logger.info(f"  Interpolated frames: {metrics['num_interpolated_frames']}")
        
        # Visualize
        fig = visualize_interpolation(series, interpolated, 
                                     f"Synthetic Test (max_gap={max_gap}s)")
        plt.savefig(f'debug_interpolation_synthetic_gap{max_gap}.png', dpi=150)
        logger.info(f"  Saved plot: debug_interpolation_synthetic_gap{max_gap}.png")
    
    plt.show()


def test_real_data(experiment: str, session: str, sensor_id: Optional[str] = None):
    """Test interpolation on real experimental data."""
    logger.info(f"Testing interpolation on {experiment} ({session})")
    
    # Load WP-2 results
    wp2_dir = PROCESSED_DATA_DIR / "rpm" / "wp2" / session
    
    if sensor_id:
        sensors = [sensor_id]
    else:
        # Find all available sensors
        h5_files = list(wp2_dir.glob(f"{experiment}_*_rpm.h5"))
        sensors = []
        for f in h5_files:
            # Extract sensor ID from filename
            parts = f.stem.split('_')
            for i, part in enumerate(parts):
                if part.startswith('Sensor'):
                    sensor = f"{parts[i]}_{parts[i+1]}"
                    sensors.append(sensor)
                    break
    
    if not sensors:
        logger.error(f"No WP-2 results found for {experiment}")
        return
    
    logger.info(f"Found sensors: {sensors}")
    
    # Process each sensor
    for sensor in sensors:
        logger.info(f"\nProcessing {sensor}...")
        
        # Load data
        h5_file = wp2_dir / f"{experiment}_{sensor}_rpm.h5" / f"{experiment}_{sensor}_rpm.h5"
        
        if not h5_file.exists():
            logger.warning(f"File not found: {h5_file}")
            continue
        
        # Load time series from HDF5
        import h5py
        
        with h5py.File(h5_file, 'r') as f:
            if 'time' not in f or 'rpm' not in f or 'snr_db' not in f:
                logger.error(f"Invalid HDF5 structure in {h5_file}")
                continue
            
            times = f['time'][:]
            rpms = f['rpm'][:]
            snrs = f['snr_db'][:]
        
        # Create RPMTimeSeries
        frames = []
        for i in range(len(times)):
            frame = RPMFrame(
                time=times[i],
                rpm=rpms[i],
                snr_db=snrs[i],
                sensor_id=sensor,
                method='welch',
                confidence=0.8
            )
            frames.append(frame)
        
        series = RPMTimeSeries(
            frames=frames,
            experiment=experiment,
            session=session,
            sensor_id=sensor
        )
        
        logger.info(f"  Loaded {len(frames)} frames, {series.availability:.1f}% availability")
        
        # Test interpolation with different settings
        for max_gap in [3.0, 5.0, 10.0]:
            interpolated = interpolate_missing_frames(series, max_gap_s=max_gap)
            
            metrics = analyze_interpolation_quality(series, interpolated)
            
            logger.info(f"  max_gap={max_gap}s: {metrics['original_availability']:.1f}% -> "
                       f"{metrics['interpolated_availability']:.1f}% "
                       f"(+{metrics['availability_gain']:.1f}%)")
        
        # Visualize best result
        interpolated = interpolate_missing_frames(series, max_gap_s=5.0)
        fig = visualize_interpolation(series, interpolated, 
                                     f"{experiment} - {sensor} (max_gap=5s)")
        
        output_file = f'debug_interpolation_{experiment}_{sensor}.png'
        plt.savefig(output_file, dpi=150)
        logger.info(f"  Saved plot: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Debug RPM interpolation')
    
    parser.add_argument('--test-synthetic', action='store_true',
                       help='Run synthetic data test')
    parser.add_argument('--experiment', type=str,
                       help='Experiment name for real data test')
    parser.add_argument('--session', choices=['morning', 'afternoon'],
                       help='Session for real data test')
    parser.add_argument('--sensor', type=str,
                       help='Specific sensor ID (optional)')
    parser.add_argument('--plot', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--max-gap', type=float, default=5.0,
                       help='Maximum gap to interpolate (seconds)')
    
    args = parser.parse_args()
    
    if args.test_synthetic:
        test_synthetic()
    elif args.experiment and args.session:
        test_real_data(args.experiment, args.session, args.sensor)
    else:
        # Run default synthetic test
        logger.info("No arguments provided, running synthetic test...")
        test_synthetic()
    
    if args.plot:
        plt.show()


if __name__ == '__main__':
    main()