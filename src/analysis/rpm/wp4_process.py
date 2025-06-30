"""
WP-4: Multi-sensor fusion processing module.

This module implements the main processing pipeline for WP-4,
combining RPM estimates from multiple sensors using confidence-based fusion.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import h5py
from dataclasses import dataclass
import json
from datetime import datetime

from .io import load_config, find_experiment_data
from .tracking import RPMFrame, RPMTimeSeries
from .fusion import (
    fuse_sensors_snr, 
    interpolate_missing_frames,
    apply_median_filter,
    compute_sensor_agreement
)
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Container for fusion processing results."""
    fused_series: RPMTimeSeries
    sensor_contributions: Dict[str, float]
    quality_stats: Dict[str, float]
    method_usage: Dict[str, float]
    processing_time: float


def load_wp2_results(experiment: str, session: str, sensor_id: str, 
                     base_path: Path) -> Optional[RPMTimeSeries]:
    """Load WP-2 (Welch) results from HDF5 file."""
    wp2_path = base_path / "results" / "wp2" / session
    h5_file = wp2_path / f"{experiment}_{sensor_id}_rpm.h5" / f"{experiment}_{sensor_id}_rpm.h5"
    
    if not h5_file.exists():
        logger.warning(f"WP-2 results not found: {h5_file}")
        return None
    
    try:
        frames = []
        with h5py.File(h5_file, 'r') as f:
            # Load time series data from rpm_estimation group
            grp = f['rpm_estimation'] if 'rpm_estimation' in f else f
            times = grp['time'][:]
            rpms = grp['rpm'][:]
            snrs = grp['snr_db'][:]
            
            # Load metadata if available
            metadata = {}
            if 'metadata' in f.attrs:
                metadata = json.loads(f.attrs['metadata'])
            
            # Create RPMFrame objects
            for t, rpm, snr in zip(times, rpms, snrs):
                if not np.isnan(rpm) and not np.isnan(snr):
                    frame = RPMFrame(
                        time=float(t),
                        rpm=float(rpm),
                        snr_db=float(snr),
                        sensor_id=sensor_id,
                        method='welch',
                        confidence=1.0 if snr >= 10 else 0.5
                    )
                    frames.append(frame)
        
        series = RPMTimeSeries(
            frames=frames,
            experiment=experiment,
            session=session,
            sensor_id=sensor_id
        )
        
        logger.info(f"Loaded WP-2 data: {len(frames)} frames, "
                   f"{series.availability:.1f}% availability")
        return series
        
    except Exception as e:
        logger.error(f"Error loading WP-2 results: {e}")
        return None


def load_wp3_results(experiment: str, session: str, sensor_id: str,
                     base_path: Path) -> Optional[RPMTimeSeries]:
    """Load WP-3 (STFT) results from HDF5 file."""
    wp3_path = base_path / "results" / "wp3" / session
    h5_file = wp3_path / f"{experiment}_{sensor_id}_stft.h5"
    
    if not h5_file.exists():
        logger.warning(f"WP-3 results not found: {h5_file}")
        return None
    
    try:
        frames = []
        with h5py.File(h5_file, 'r') as f:
            # Load time series data from data group
            if 'data' in f:
                data_grp = f['data']
                times = data_grp['time'][:]
                rpms = data_grp['rpm_est'][:] if 'rpm_est' in data_grp else data_grp['rpm'][:]
                snrs = data_grp['snr_db'][:]
                
                # Check for smoothed data
                if 'smoothed_rpm' in data_grp:
                    rpms_smooth = data_grp['smoothed_rpm'][:]
                    use_smoothed = True
                elif 'rpm_smoothed' in data_grp:
                    rpms_smooth = data_grp['rpm_smoothed'][:]
                    use_smoothed = True
                else:
                    rpms_smooth = rpms
                    use_smoothed = False
            else:
                # Fallback to root level
                times = f['time'][:]
                rpms = f['rpm'][:]
                snrs = f['snr_db'][:]
                rpms_smooth = rpms
                use_smoothed = False
            
            # Create RPMFrame objects
            for i, (t, rpm, snr) in enumerate(zip(times, rpms_smooth, snrs)):
                if not np.isnan(rpm) and not np.isnan(snr):
                    frame = RPMFrame(
                        time=float(t),
                        rpm=float(rpm),
                        snr_db=float(snr),
                        sensor_id=sensor_id,
                        method='stft' if not use_smoothed else 'stft_smoothed',
                        confidence=1.0 if snr >= 10 else 0.5
                    )
                    frames.append(frame)
        
        series = RPMTimeSeries(
            frames=frames,
            experiment=experiment,
            session=session,
            sensor_id=sensor_id
        )
        
        logger.info(f"Loaded WP-3 data: {len(frames)} frames, "
                   f"{series.availability:.1f}% availability")
        return series
        
    except Exception as e:
        logger.error(f"Error loading WP-3 results: {e}")
        return None


def select_method_for_region(wp2_series: Optional[RPMTimeSeries],
                           wp3_series: Optional[RPMTimeSeries],
                           start_time: float,
                           end_time: float,
                           config: dict) -> RPMTimeSeries:
    """
    Select appropriate method (Welch or STFT) for a time region.
    
    Decision based on:
    - Rate of change (prefer STFT for transients)
    - Data availability
    - SNR quality
    """
    if wp2_series is None:
        return wp3_series if wp3_series else RPMTimeSeries([], "", "", "")
    if wp3_series is None:
        return wp2_series
    
    # Extract frames in time window
    wp2_frames = [f for f in wp2_series.frames 
                  if start_time <= f.time <= end_time and f.is_valid()]
    wp3_frames = [f for f in wp3_series.frames 
                  if start_time <= f.time <= end_time and f.is_valid()]
    
    if not wp2_frames:
        return wp3_series
    if not wp3_frames:
        return wp2_series
    
    # Calculate rate of change
    if len(wp3_frames) > 2:
        times = np.array([f.time for f in wp3_frames])
        rpms = np.array([f.rpm for f in wp3_frames])
        
        # Simple derivative estimate
        dt = np.mean(np.diff(times))
        if dt > 0:
            drpm_dt = np.abs(np.gradient(rpms, times))
            max_rate = np.max(drpm_dt)
            
            # Use STFT if rate exceeds threshold
            threshold = config.get('wp4', {}).get('method_selection', {}).get(
                'steady_state_threshold', 50)
            
            if max_rate > threshold:
                logger.debug(f"High rate detected ({max_rate:.1f} RPM/s), using STFT")
                return wp3_series
    
    # Default to Welch for steady-state (better frequency resolution)
    avg_snr_wp2 = np.mean([f.snr_db for f in wp2_frames])
    avg_snr_wp3 = np.mean([f.snr_db for f in wp3_frames])
    
    if avg_snr_wp2 >= avg_snr_wp3:
        return wp2_series
    else:
        return wp3_series


def process_experiment_fusion(experiment: str, session: str, config: dict,
                            base_path: Path) -> FusionResult:
    """
    Process a single experiment with multi-sensor fusion.
    
    This is the main entry point for WP-4 processing.
    """
    start_time = datetime.now()
    
    # Get sensor list from config
    sensors = config.get('sensors', {}).get('primary', 
                                           ['Sensor_3', 'Sensor_4', 'Sensor_wb'])
    
    logger.info(f"Processing fusion for {experiment} ({session}) with sensors: {sensors}")
    
    # Load all sensor data
    sensor_data_wp2 = {}
    sensor_data_wp3 = {}
    
    for sensor in sensors:
        # Load WP-2 results
        wp2_series = load_wp2_results(experiment, session, sensor, base_path)
        if wp2_series and len(wp2_series.frames) > 0:
            sensor_data_wp2[sensor] = wp2_series
        
        # Load WP-3 results
        wp3_series = load_wp3_results(experiment, session, sensor, base_path)
        if wp3_series and len(wp3_series.frames) > 0:
            sensor_data_wp3[sensor] = wp3_series
    
    if not sensor_data_wp2 and not sensor_data_wp3:
        raise ValueError(f"No data found for experiment {experiment}")
    
    # Combine WP-2 and WP-3 data with intelligent method selection
    combined_sensor_data = {}
    method_usage = {'welch': 0, 'stft': 0, 'stft_smoothed': 0}
    
    for sensor in sensors:
        wp2 = sensor_data_wp2.get(sensor)
        wp3 = sensor_data_wp3.get(sensor)
        
        if wp2 or wp3:
            # For now, prefer WP-3 (STFT) if available due to better time resolution
            # In future, implement dynamic selection based on rate of change
            if wp3 and wp3.availability > 50:
                combined_sensor_data[sensor] = wp3
                method_usage['stft'] += len(wp3.frames)
            elif wp2:
                combined_sensor_data[sensor] = wp2
                method_usage['welch'] += len(wp2.frames)
    
    if not combined_sensor_data:
        raise ValueError("No valid sensor data after method selection")
    
    # Apply multi-sensor fusion
    fusion_config = config.get('wp4', {}).get('fusion', {})
    fused_series = fuse_sensors_snr(combined_sensor_data, fusion_config)
    
    # Apply interpolation for gaps
    interp_config = config.get('wp4', {}).get('interpolation', {})
    max_gap = interp_config.get('max_gap_s', 5.0)
    fused_series = interpolate_missing_frames(fused_series, max_gap)
    
    # Apply median filter for outlier removal
    fused_series = apply_median_filter(fused_series, window_s=1.0)
    
    # Calculate sensor contributions
    sensor_contributions = {}
    total_frames = len(fused_series.frames)
    
    for sensor in sensors:
        sensor_frames = [f for f in fused_series.frames 
                        if sensor in f.sensor_id]
        sensor_contributions[sensor] = len(sensor_frames) / total_frames if total_frames > 0 else 0
    
    # Calculate quality statistics
    valid_frames = [f for f in fused_series.frames if f.is_valid()]
    interp_frames = [f for f in fused_series.frames if f.method == 'interpolated']
    high_snr_frames = [f for f in valid_frames if f.snr_db >= 15]
    
    quality_stats = {
        'availability': fused_series.availability,
        'interpolated_fraction': len(interp_frames) / total_frames if total_frames > 0 else 0,
        'high_snr_fraction': len(high_snr_frames) / len(valid_frames) if valid_frames else 0,
        'mean_snr_db': np.mean([f.snr_db for f in valid_frames]) if valid_frames else 0,
        'sensor_agreement_mean': np.mean([f.confidence for f in valid_frames]) if valid_frames else 0
    }
    
    # Normalize method usage
    total_method_frames = sum(method_usage.values())
    if total_method_frames > 0:
        method_usage = {k: v/total_method_frames for k, v in method_usage.items()}
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    result = FusionResult(
        fused_series=fused_series,
        sensor_contributions=sensor_contributions,
        quality_stats=quality_stats,
        method_usage=method_usage,
        processing_time=processing_time
    )
    
    logger.info(f"Fusion complete: {fused_series.availability:.1f}% availability, "
               f"{quality_stats['mean_snr_db']:.1f} dB mean SNR")
    
    return result


def save_fusion_results(result: FusionResult, output_path: Path, config: dict):
    """Save fusion results to CSV and generate report."""
    
    # Save fused time series as CSV
    times, rpms, snrs = result.fused_series.to_arrays()
    
    # Build DataFrame with all relevant columns
    df_data = {
        'time': times,
        'rpm': rpms,
        'snr_db': snrs,
        'sensor_id': [f.sensor_id for f in result.fused_series.frames],
        'method': [f.method for f in result.fused_series.frames],
        'quality': ['measured' if f.is_valid() else 'interpolated' 
                   for f in result.fused_series.frames],
        'rpm_valid': [f.is_valid() for f in result.fused_series.frames]
    }
    
    df = pd.DataFrame(df_data)
    csv_path = output_path / "rpm_fused.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved fused RPM to {csv_path}")
    
    # Generate fusion report
    report = {
        'experiment': result.fused_series.experiment,
        'session': result.fused_series.session,
        'processing_time_s': result.processing_time,
        'total_frames': len(result.fused_series.frames),
        'sensor_contributions': result.sensor_contributions,
        'quality_statistics': result.quality_stats,
        'method_usage': result.method_usage,
        'time_range': {
            'start': float(times[0]) if len(times) > 0 else 0,
            'end': float(times[-1]) if len(times) > 0 else 0,
            'duration_s': float(times[-1] - times[0]) if len(times) > 0 else 0
        }
    }
    
    report_path = output_path / "fusion_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved fusion report to {report_path}")
    
    # Log summary statistics
    logger.info(f"Fusion summary for {result.fused_series.experiment}:")
    logger.info(f"  - Availability: {result.quality_stats['availability']:.1f}%")
    logger.info(f"  - Mean SNR: {result.quality_stats['mean_snr_db']:.1f} dB")
    logger.info(f"  - Interpolated: {result.quality_stats['interpolated_fraction']*100:.1f}%")
    logger.info(f"  - Sensor contributions: {result.sensor_contributions}")


def create_diagnostic_plots(result: FusionResult, output_path: Path):
    """Create diagnostic visualizations for fusion results."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    times, rpms, snrs = result.fused_series.to_arrays()
    
    if len(times) == 0:
        logger.warning("No data to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. RPM time series with quality indicators
    ax1 = axes[0]
    
    # Color by quality
    valid_mask = np.array([f.is_valid() for f in result.fused_series.frames])
    
    # Plot valid and interpolated separately
    ax1.scatter(times[valid_mask], rpms[valid_mask], 
               c='blue', s=10, alpha=0.6, label='Measured')
    ax1.scatter(times[~valid_mask], rpms[~valid_mask], 
               c='red', s=10, alpha=0.6, label='Interpolated')
    
    ax1.set_ylabel('RPM')
    ax1.set_title(f'Fused RPM - {result.fused_series.experiment} ({result.fused_series.session})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. SNR and sensor selection
    ax2 = axes[1]
    
    # Plot SNR
    ax2.plot(times, snrs, 'g-', linewidth=1, label='SNR')
    ax2.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Threshold')
    
    ax2.set_ylabel('SNR (dB)')
    ax2.set_ylim(0, max(30, np.percentile(snrs[valid_mask], 95) if np.any(valid_mask) else 20))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Sensor contribution over time
    ax3 = axes[2]
    
    # Create sensor ID mapping
    sensor_colors = {
        'Sensor_3': 'blue',
        'Sensor_4': 'green', 
        'Sensor_wb': 'orange',
        'fused': 'purple'
    }
    
    # Plot sensor usage as colored bars
    for i, frame in enumerate(result.fused_series.frames):
        if frame.is_valid():
            # Extract base sensor from fused ID
            base_sensor = frame.sensor_id.replace('fused_', '')
            color = sensor_colors.get(base_sensor, 'gray')
            ax3.barh(0, times[i+1]-times[i] if i < len(times)-1 else 0.25, 
                    left=times[i], height=0.8, color=color, alpha=0.8)
    
    # Create legend
    patches = [mpatches.Patch(color=color, label=sensor) 
              for sensor, color in sensor_colors.items() if sensor != 'fused']
    ax3.legend(handles=patches, loc='upper right')
    
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_yticks([])
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Sensor Selection Timeline')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / "fusion_diagnostic.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved diagnostic plot to {plot_path}")


def main(experiment: str, session: str, config_path: Path,
         output_dir: Optional[Path] = None, plot: bool = True):
    """Main entry point for WP-4 processing."""
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up paths
    base_path = Path(__file__).parent
    if output_dir is None:
        output_dir = base_path / "results" / "wp4" / session / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process experiment
    try:
        result = process_experiment_fusion(experiment, session, config, base_path)
        
        # Save results
        save_fusion_results(result, output_dir, config)
        
        # Create plots if requested
        if plot:
            create_diagnostic_plots(result, output_dir)
        
        # Check success criteria
        if result.quality_stats['availability'] < 95:
            logger.warning(f"Availability {result.quality_stats['availability']:.1f}% "
                          f"below 95% target")
        
        nan_fraction = 1 - result.quality_stats['availability'] / 100
        if nan_fraction > 0.02:
            logger.warning(f"NaN fraction {nan_fraction*100:.1f}% exceeds 2% target")
        else:
            logger.info(f"Success: NaN fraction {nan_fraction*100:.1f}% meets <2% target")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {experiment}: {e}")
        raise


if __name__ == "__main__":
    # Test with example experiment
    setup_logging(log_file=Path("wp4_test.log"))
    
    experiment = "026_Engine_rpm_sweep"
    session = "afternoon"
    config_path = Path("rpm_config.yaml")
    
    result = main(experiment, session, config_path, plot=True)