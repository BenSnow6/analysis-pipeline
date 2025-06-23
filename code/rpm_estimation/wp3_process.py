#!/usr/bin/env python3
"""
WP-3: STFT-based RPM extraction with time resolution and quality control.

This script processes aligned IMU data to extract time-resolved RPM estimates
using Short-Time Fourier Transform (STFT) with early SNR gating and optional
smoothing for high-rate transients.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
try:
    from .logging_config import setup_logging
    from .io import load_aligned_data, load_config
    from .spectral import extract_rpm_stft
    from .tracking import RPMTimeSeries, smooth_rpm_series
    from .quality import verify_antialiasing_filter, load_qa_summary
except ImportError:
    # For direct execution
    from logging_config import setup_logging
    # First import the module with a different name
    import io as io_module
    from spectral import extract_rpm_stft
    from tracking import RPMTimeSeries, smooth_rpm_series
    from quality import verify_antialiasing_filter, load_qa_summary
    
    # Get functions from our io module
    load_aligned_data = io_module.load_aligned_data
    load_config = io_module.load_config


def process_single_sensor(
    data_path: Path,
    qa_path: Path,
    sensor_id: str,
    config: dict,
    experiment: str,
    session: str,
    output_dir: Path
) -> Tuple[bool, Optional[Path]]:
    """
    Process a single sensor's data with STFT-based RPM extraction.
    
    Args:
        data_path: Path to proc_IMU parquet file
        qa_path: Path to QA summary JSON
        sensor_id: Sensor identifier
        config: Configuration dictionary
        experiment: Experiment name
        session: Session type (morning/afternoon)
        output_dir: Output directory
        
    Returns:
        Tuple of (success, output_path)
    """
    logger = logging.getLogger(__name__)
    
    # Load and verify QA summary
    qa_summary = load_qa_summary(qa_path)
    if qa_summary is None:
        logger.error(f"Failed to load QA summary for {sensor_id}")
        return False, None
    
    # Verify anti-aliasing filter
    filter_verified, filter_details = verify_antialiasing_filter(qa_summary, config)
    
    require_antialiasing = config.get('wp3', {}).get('quality', {}).get('require_antialiasing', True)
    if require_antialiasing and not filter_verified:
        logger.error(
            f"Anti-aliasing filter verification failed for {sensor_id}",
            extra={'warnings': filter_details.get('warnings', [])}
        )
        return False, None
    
    # Load processed data
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path.name}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False, None
    
    # Extract vibration magnitude
    if 'a_hp_mag' not in df.columns:
        logger.error("Vibration magnitude (a_hp_mag) not found in data")
        return False, None
    
    vibration_mag = df['a_hp_mag'].values
    time = df['time_from_sync'].values
    
    # Check minimum data length
    min_seconds = config.get('wp3', {}).get('processing', {}).get('min_data_seconds', 2.0)
    if len(time) < min_seconds * config['fs']:
        logger.warning(f"Insufficient data: {len(time)/config['fs']:.1f}s < {min_seconds}s")
        return False, None
    
    # Extract RPM using STFT
    logger.info("Starting STFT-based RPM extraction")
    rpm_series = extract_rpm_stft(
        vibration_mag,
        fs=config['fs'],
        config=config,
        start_time=time[0],
        sensor_id=sensor_id
    )
    
    # Update metadata
    rpm_series.experiment = experiment
    rpm_series.session = session
    rpm_series.metadata.update({
        'filter_verified': filter_verified,
        'filter_details': filter_details,
        'processing_timestamp': datetime.utcnow().isoformat() + 'Z'
    })
    
    # Apply smoothing if enabled
    smoothing_config = config.get('wp3', {}).get('smoothing', {})
    if smoothing_config.get('enabled', True):
        times, rpms, _ = rpm_series.to_arrays()
        
        smoothed_rpm = smooth_rpm_series(
            times, rpms,
            method=smoothing_config.get('method', 'polynomial'),
            window=smoothing_config.get('window_size', 5),
            high_rate_threshold=smoothing_config.get('high_rate_threshold', 150.0)
        )
        
        # Add smoothed values to frames
        for i, frame in enumerate(rpm_series.frames):
            frame.metadata['smoothed_rpm'] = float(smoothed_rpm[i])
    
    # Save results
    output_path = output_dir / f"{experiment}_{sensor_id}_stft.h5"
    success = save_rpm_series_hdf5(rpm_series, output_path, config)
    
    if success:
        logger.info(
            f"Saved STFT results to {output_path}",
            extra={
                'availability': rpm_series.availability,
                'valid_frames': len(rpm_series.get_valid_frames()),
                'total_frames': len(rpm_series.frames)
            }
        )
        return True, output_path
    else:
        return False, None


def save_rpm_series_hdf5(rpm_series: RPMTimeSeries, 
                        output_path: Path,
                        config: dict) -> bool:
    """
    Save RPM time series to HDF5 file with metadata.
    
    Args:
        rpm_series: RPM time series object
        output_path: Output file path
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as hf:
            # Create metadata group
            meta_grp = hf.create_group('metadata')
            meta_grp.attrs['experiment'] = rpm_series.experiment or ''
            meta_grp.attrs['session'] = rpm_series.session or ''
            meta_grp.attrs['sensor_id'] = rpm_series.sensor_id or ''
            meta_grp.attrs['method'] = 'stft'
            meta_grp.attrs['processing_timestamp'] = rpm_series.metadata.get(
                'processing_timestamp', datetime.utcnow().isoformat() + 'Z'
            )
            
            # Store filter verification
            if 'filter_verified' in rpm_series.metadata:
                meta_grp.attrs['anti_alias_verified'] = rpm_series.metadata['filter_verified']
            
            # Store STFT parameters
            if 'stft_params' in rpm_series.metadata:
                for key, value in rpm_series.metadata['stft_params'].items():
                    meta_grp.attrs[f'stft_{key}'] = value
            
            # Store config
            meta_grp.attrs['config'] = json.dumps({
                'wp3': config.get('wp3', {}),
                'fs': config.get('fs', 200),
                'snr': config.get('snr', {})
            })
            
            # Create data group
            data_grp = hf.create_group('data')
            
            # Extract arrays
            times, rpms, snrs = rpm_series.to_arrays()
            
            # Store time series
            data_grp.create_dataset('time', data=times)
            data_grp.create_dataset('rpm_est', data=rpms)
            data_grp.create_dataset('snr_db', data=snrs)
            
            # Store validity flags
            valid = np.array([not np.isnan(f.rpm) for f in rpm_series.frames])
            data_grp.create_dataset('valid', data=valid)
            
            # Store confidence scores if available
            confidence = np.array([f.confidence if f.confidence is not None else 
                                 (1.0 if not np.isnan(f.rpm) else 0.0) 
                                 for f in rpm_series.frames])
            data_grp.create_dataset('confidence', data=confidence)
            
            # Store smoothed RPM if available
            if rpm_series.frames and 'smoothed_rpm' in rpm_series.frames[0].metadata:
                smoothed = np.array([f.metadata.get('smoothed_rpm', np.nan) 
                                   for f in rpm_series.frames])
                data_grp.create_dataset('smoothed_rpm', data=smoothed)
            
            # Create quality group
            quality_grp = hf.create_group('quality')
            quality_grp.attrs['availability'] = rpm_series.availability
            quality_grp.attrs['mean_snr'] = float(np.nanmean(snrs))
            
            # Calculate max delta RPM
            valid_rpms = rpms[~np.isnan(rpms)]
            if len(valid_rpms) > 1:
                valid_times = times[~np.isnan(rpms)]
                dt = np.diff(valid_times)
                drpm = np.abs(np.diff(valid_rpms))
                max_delta_rpm = float(np.max(drpm / dt)) if len(dt) > 0 else 0.0
            else:
                max_delta_rpm = 0.0
            quality_grp.attrs['max_delta_rpm'] = max_delta_rpm
            
        logger.debug(f"Successfully saved HDF5 to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save HDF5: {e}")
        return False


def generate_diagnostic_plot(rpm_series: RPMTimeSeries,
                           output_path: Path,
                           experiment: str,
                           sensor_id: str) -> None:
    """Generate diagnostic plot for STFT results."""
    logger = logging.getLogger(__name__)
    
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        times, rpms, snrs = rpm_series.to_arrays()
        valid_mask = ~np.isnan(rpms)
        
        # Plot 1: RPM over time
        ax1 = axes[0]
        if np.any(valid_mask):
            ax1.scatter(times[valid_mask], rpms[valid_mask], 
                       c='blue', s=20, alpha=0.7, label='Valid')
        if np.any(~valid_mask):
            ax1.scatter(times[~valid_mask], np.zeros(np.sum(~valid_mask)), 
                       c='red', marker='x', s=20, alpha=0.5, label='Gated')
        
        # Add smoothed line if available
        if rpm_series.frames and 'smoothed_rpm' in rpm_series.frames[0].metadata:
            smoothed = np.array([f.metadata.get('smoothed_rpm', np.nan) 
                               for f in rpm_series.frames])
            smoothed_valid = ~np.isnan(smoothed)
            if np.any(smoothed_valid):
                ax1.plot(times[smoothed_valid], smoothed[smoothed_valid], 
                        'g-', linewidth=2, alpha=0.8, label='Smoothed')
        
        ax1.set_ylabel('RPM')
        ax1.set_title(f'STFT RPM Extraction - {experiment} - {sensor_id}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: SNR over time
        ax2 = axes[1]
        ax2.plot(times, snrs, 'b-', linewidth=1)
        ax2.axhline(y=10.0, color='r', linestyle='--', label='Threshold')
        ax2.set_ylabel('SNR (dB)')
        ax2.set_ylim([0, max(20, np.max(snrs[~np.isnan(snrs)]) * 1.1)])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: RPM rate of change
        ax3 = axes[2]
        if np.sum(valid_mask) > 1:
            valid_times = times[valid_mask]
            valid_rpms = rpms[valid_mask]
            dt = np.diff(valid_times)
            drpm = np.diff(valid_rpms)
            rpm_rate = drpm / dt
            
            ax3.plot(valid_times[1:], rpm_rate, 'b-', linewidth=1)
            ax3.axhline(y=150, color='r', linestyle='--', label='High-rate threshold')
            ax3.axhline(y=-150, color='r', linestyle='--')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('RPM Rate (RPM/s)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add summary text
        availability = rpm_series.availability
        mean_rpm = rpm_series.mean_rpm
        text = f"Availability: {availability:.1f}%\nMean RPM: {mean_rpm:.0f}"
        ax1.text(0.02, 0.98, text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved diagnostic plot to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}")


def process_experiment(experiment: str, session: str, 
                      sensors: Optional[List[str]] = None,
                      config_path: Optional[Path] = None,
                      generate_plots: bool = True) -> Dict[str, Path]:
    """
    Process one experiment with all specified sensors.
    
    Args:
        experiment: Experiment name
        session: Session type (morning/afternoon)
        sensors: List of sensors to process (None for all)
        config_path: Path to config file
        generate_plots: Whether to generate diagnostic plots
        
    Returns:
        Dictionary mapping sensor_id to output path
    """
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent / "rpm_config.yaml"
    
    config = load_config(config_path)
    
    # Determine sensors to process
    if sensors is None:
        sensors = config.get('wp1', {}).get('sensors', {}).get('default', 
                           ['Sensor_3', 'Sensor_4', 'Sensor_wb'])
    
    logger.info(
        f"Processing experiment {experiment} ({session})",
        extra={'sensors': sensors}
    )
    
    # Setup paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / 'results' / 'wp1' / 'output_wp1' / session / experiment
    output_dir = base_dir / 'results' / 'wp3' / session
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sensor
    results = {}
    for sensor_id in sensors:
        logger.info(f"Processing sensor {sensor_id}")
        
        # Check input files
        data_path = input_dir / f"proc_IMU_{sensor_id}.parquet"
        qa_path = input_dir / f"qa_summary_{sensor_id}.json"
        
        if not data_path.exists():
            logger.warning(f"Data file not found: {data_path}")
            continue
            
        if not qa_path.exists():
            logger.warning(f"QA file not found: {qa_path}")
            continue
        
        # Process sensor
        success, output_path = process_single_sensor(
            data_path, qa_path, sensor_id, config,
            experiment, session, output_dir
        )
        
        if success and output_path:
            results[sensor_id] = output_path
            
            # Generate diagnostic plot
            if generate_plots:
                plot_dir = output_dir.parent / 'plots' / session
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plot_dir / f"{experiment}_{sensor_id}_stft_diagnostic.png"
                
                # Load results for plotting
                with h5py.File(output_path, 'r') as hf:
                    # Recreate RPMTimeSeries for plotting
                    from tracking import RPMFrame
                    times = hf['data/time'][:]
                    rpms = hf['data/rpm_est'][:]
                    snrs = hf['data/snr_db'][:]
                    
                    frames = []
                    for i in range(len(times)):
                        frame = RPMFrame(
                            time=times[i],
                            rpm=rpms[i],
                            snr_db=snrs[i],
                            sensor_id=sensor_id,
                            method='stft'
                        )
                        if 'smoothed_rpm' in hf['data']:
                            frame.metadata['smoothed_rpm'] = hf['data/smoothed_rpm'][i]
                        frames.append(frame)
                    
                    plot_series = RPMTimeSeries(
                        frames=frames,
                        experiment=experiment,
                        session=session,
                        sensor_id=sensor_id
                    )
                    
                generate_diagnostic_plot(plot_series, plot_path, experiment, sensor_id)
    
    logger.info(
        f"Completed processing {experiment}",
        extra={'successful_sensors': len(results)}
    )
    
    return results


def main():
    """Main entry point for WP-3 processing."""
    parser = argparse.ArgumentParser(
        description="WP-3: STFT-based RPM extraction with quality control"
    )
    
    parser.add_argument(
        '--experiment', '-e',
        required=True,
        help='Experiment name (e.g., 026_Engine_rpm_sweep)'
    )
    parser.add_argument(
        '--session', '-s',
        required=True,
        choices=['morning', 'afternoon'],
        help='Session type'
    )
    parser.add_argument(
        '--sensors',
        nargs='+',
        help='Sensors to process (default: from config)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip diagnostic plot generation'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all experiments in batch mode'
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing - implement later
        print("Batch processing not yet implemented")
        sys.exit(1)
    else:
        # Single experiment
        results = process_experiment(
            args.experiment,
            args.session,
            args.sensors,
            args.config,
            generate_plots=not args.no_plots
        )
        
        if results:
            print(f"\nSuccessfully processed {len(results)} sensors:")
            for sensor_id, path in results.items():
                print(f"  {sensor_id}: {path}")
        else:
            print("\nNo sensors processed successfully")
            sys.exit(1)


if __name__ == '__main__':
    main()