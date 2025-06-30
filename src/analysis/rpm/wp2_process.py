"""
WP-2: Welch PSD Core Processing Script

This script processes aligned vibration data from WP-1 to extract RPM estimates
using Welch PSD analysis.
"""

import argparse
import logging
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
# Import RPM estimation modules
try:
    from .spectral import extract_rpm_from_vibration, welch_psd, find_peaks_in_psd
    from .tracking import RPMFrame, RPMTimeSeries
except ImportError:
    from src.analysis.rpm.spectral import extract_rpm_from_vibration, welch_psd, find_peaks_in_psd
    from src.analysis.rpm.tracking import RPMFrame, RPMTimeSeries


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_processed_data(experiment: str, sensor_id: str, session: str, 
                       base_path: Path) -> Optional[pd.DataFrame]:
    """Load processed parquet file from WP-1."""
    # Construct path to aligned data
    aligned_path = base_path / "hovercraft_data_analysis" / "alignment_analysis" / "aligned_data"
    
    # Check different possible locations
    possible_paths = [
        aligned_path / session / f"{experiment}_csv" / f"{sensor_id}.csv",
        aligned_path / f"{experiment}_csv" / f"{sensor_id}.csv",
    ]
    
    for parquet_path in possible_paths:
        if parquet_path.exists():
            logging.info(f"Loading data from: {parquet_path}")
            try:
                # Load CSV data (aligned data is in CSV format)
                df = pd.read_csv(parquet_path)
                return df
            except Exception as e:
                logging.error(f"Failed to load {parquet_path}: {e}")
                return None
    
    logging.warning(f"No data found for {experiment}/{sensor_id} in {session}")
    return None


def process_windowed_rpm(data: pd.DataFrame, config: dict, sensor_id: str,
                        window_seconds: float = 30.0, 
                        hop_seconds: float = 15.0) -> List[RPMFrame]:
    """
    Process data in windows to extract time-varying RPM.
    
    Args:
        data: DataFrame with vibration data
        config: Configuration dictionary
        sensor_id: Sensor identifier
        window_seconds: Window size in seconds
        hop_seconds: Hop size in seconds
        
    Returns:
        List of RPMFrame objects
    """
    fs = config['fs']
    window_samples = int(window_seconds * fs)
    hop_samples = int(hop_seconds * fs)
    
    # Check if we have the required columns
    if 'a_hp_mag' in data.columns:
        vibration_mag = data['a_hp_mag'].values
    else:
        # Compute magnitude from components
        a_x = data.get('x_body', data.get('ax', 0)).values
        a_y = data.get('y_body', data.get('ay', 0)).values
        a_z = data.get('z_body', data.get('az', 0)).values
        vibration_mag = np.sqrt(a_x**2 + a_y**2 + a_z**2)
    
    times = data['time_from_sync'].values
    
    rpm_frames = []
    
    # Process in overlapping windows
    for start_idx in range(0, len(vibration_mag) - window_samples + 1, hop_samples):
        end_idx = start_idx + window_samples
        
        # Extract window
        window_data = vibration_mag[start_idx:end_idx]
        window_time = times[start_idx + window_samples // 2]  # Center time
        
        # Extract RPM
        rpm_frame = extract_rpm_from_vibration(
            window_data, fs, config, 
            timestamp=window_time, 
            sensor_id=sensor_id
        )
        
        if rpm_frame is not None:
            rpm_frames.append(rpm_frame)
            logging.debug(f"Window at t={window_time:.1f}s: "
                         f"RPM={rpm_frame.rpm:.1f}, SNR={rpm_frame.snr_db:.1f} dB")
    
    return rpm_frames


def save_rpm_results(rpm_series: RPMTimeSeries, output_dir: Path, 
                    save_psd: bool = True) -> Path:
    """Save RPM results to HDF5 file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{rpm_series.experiment}_{rpm_series.sensor_id}_rpm.h5"
    output_path = output_dir / filename
    
    with h5py.File(output_path, 'w') as f:
        # Create main group
        grp = f.create_group('rpm_estimation')
        
        # Save metadata
        grp.attrs['experiment'] = rpm_series.experiment
        grp.attrs['session'] = rpm_series.session
        grp.attrs['sensor_id'] = rpm_series.sensor_id
        grp.attrs['method'] = 'welch'
        grp.attrs['timestamp'] = datetime.now().isoformat()
        
        # Save time series data
        times, rpms, snrs = rpm_series.to_arrays()
        grp.create_dataset('time', data=times)
        grp.create_dataset('rpm', data=rpms)
        grp.create_dataset('snr_db', data=snrs)
        
        # Save validity flags
        valid_flags = [f.is_valid() for f in rpm_series.frames]
        grp.create_dataset('valid', data=valid_flags)
        
        # Save harmonics if available
        if rpm_series.frames and hasattr(rpm_series.frames[0], 'metadata'):
            harm_grp = grp.create_group('harmonics')
            for i, frame in enumerate(rpm_series.frames):
                if 'harmonics' in frame.metadata:
                    harm_data = frame.metadata['harmonics']
                    for h_num, h_amp in harm_data.items():
                        if str(h_num) not in harm_grp:
                            harm_grp.create_dataset(str(h_num), 
                                                   shape=(len(rpm_series.frames),),
                                                   dtype='f')
                        harm_grp[str(h_num)][i] = h_amp
        
        # Save summary statistics
        stats_grp = grp.create_group('statistics')
        stats_grp.attrs['mean_rpm'] = rpm_series.mean_rpm
        stats_grp.attrs['availability_percent'] = rpm_series.availability
        stats_grp.attrs['total_frames'] = len(rpm_series.frames)
        stats_grp.attrs['valid_frames'] = len(rpm_series.get_valid_frames())
        
    logging.info(f"Saved RPM results to: {output_path}")
    return output_path


def create_diagnostic_plots(rpm_series: RPMTimeSeries, data: pd.DataFrame,
                          config: dict, output_dir: Path):
    """Create diagnostic plots for RPM estimation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    times, rpms, snrs = rpm_series.to_arrays()
    valid_mask = np.array([f.is_valid() for f in rpm_series.frames])
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: RPM over time
    ax1 = axes[0]
    ax1.scatter(times[valid_mask], rpms[valid_mask], c='blue', label='Valid', alpha=0.6)
    ax1.scatter(times[~valid_mask], rpms[~valid_mask], c='red', label='Invalid', alpha=0.6)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('RPM')
    ax1.set_title(f'RPM Estimation - {rpm_series.experiment} - {rpm_series.sensor_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SNR over time
    ax2 = axes[1]
    ax2.plot(times, snrs, 'g-', alpha=0.7)
    ax2.axhline(y=config['snr_thresh_db'], color='r', linestyle='--', 
                label=f'Threshold ({config["snr_thresh_db"]} dB)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('Signal-to-Noise Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Example PSD from middle of data
    ax3 = axes[2]
    if 'a_hp_mag' in data.columns:
        mid_idx = len(data) // 2
        window_size = int(6 * config['fs'])  # 6 second window
        
        if mid_idx + window_size < len(data):
            window_data = data['a_hp_mag'].iloc[mid_idx:mid_idx+window_size].values
            freqs, psd = welch_psd(window_data, config['fs'], 
                                  win_sec=config['welch']['win_sec'],
                                  overlap=config['welch']['overlap'])
            
            psd_db = 10 * np.log10(psd + 1e-12)
            ax3.plot(freqs, psd_db, 'b-', alpha=0.7)
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('PSD (dB)')
            ax3.set_title('Example Power Spectral Density')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 60)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"{rpm_series.experiment}_{rpm_series.sensor_id}_diagnostic.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    logging.info(f"Saved diagnostic plot to: {plot_path}")


def process_experiment(experiment: str, session: str, config: dict, 
                      base_path: Path, output_base: Path,
                      sensors: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Process a single experiment.
    
    Returns:
        Dictionary mapping sensor_id to output file path
    """
    if sensors is None:
        sensors = config['wp1']['sensors']['default']
    
    results = {}
    
    for sensor_id in sensors:
        logging.info(f"Processing {experiment}/{sensor_id} ({session})")
        
        # Load processed data from WP-1
        data = load_processed_data(experiment, sensor_id, session, base_path)
        if data is None:
            continue
        
        # Process in windows
        rpm_frames = process_windowed_rpm(data, config, sensor_id)
        
        if not rpm_frames:
            logging.warning(f"No valid RPM estimates for {experiment}/{sensor_id}")
            continue
        
        # Create time series
        rpm_series = RPMTimeSeries(
            frames=rpm_frames,
            experiment=experiment,
            session=session,
            sensor_id=sensor_id
        )
        
        # Save results
        output_dir = output_base / "wp2" / session
        output_path = save_rpm_results(rpm_series, output_dir)
        results[sensor_id] = output_path
        
        # Create diagnostic plots
        if config.get('wp2', {}).get('output', {}).get('save_psd', True):
            plot_dir = output_base / "wp2" / "plots" / session
            create_diagnostic_plots(rpm_series, data, config, plot_dir)
        
        # Log summary
        logging.info(f"  Mean RPM: {rpm_series.mean_rpm:.1f}")
        logging.info(f"  Availability: {rpm_series.availability:.1f}%")
        logging.info(f"  Valid frames: {len(rpm_series.get_valid_frames())}/{len(rpm_series.frames)}")
    
    return results


def main():
    """Main entry point for WP-2 processing."""
    parser = argparse.ArgumentParser(description="WP-2: Welch PSD RPM Extraction")
    parser.add_argument('--experiment', '-e', help='Experiment name (e.g., 007_Fast_stbd_turn_1)')
    parser.add_argument('--session', '-s', choices=['morning', 'afternoon', 'static'],
                       help='Session type')
    parser.add_argument('--config', '-c', default='rpm_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--sensors', nargs='+', help='Specific sensors to process')
    parser.add_argument('--all', action='store_true', help='Process all experiments')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log-file', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try in the same directory as script
        config_path = Path(__file__).parent / args.config
    
    config = load_config(str(config_path))
    logging.info(f"Loaded configuration from: {config_path}")
    
    # Determine base paths
    base_path = Path(__file__).parent.parent.parent
    output_base = Path(args.output) if args.output else base_path / "results"
    
    # Process experiments
    if args.all:
        # Process all experiments
        logging.info("Processing all experiments...")
        # This would require listing all experiments from the aligned data directory
        # For now, we'll just log a message
        logging.warning("--all flag not fully implemented yet")
    else:
        if not args.experiment or not args.session:
            parser.error("Either --all or both --experiment and --session are required")
        
        results = process_experiment(
            args.experiment, 
            args.session,
            config,
            base_path,
            output_base,
            args.sensors
        )
        
        # Log results
        logging.info(f"\nProcessing complete for {args.experiment}:")
        for sensor_id, path in results.items():
            logging.info(f"  {sensor_id}: {path}")
    
    # Create completion marker
    marker_path = output_base / "wp2" / "wp2_done.flag"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(f"WP-2 completed at {datetime.now().isoformat()}\n")
    logging.info(f"Created completion marker: {marker_path}")


if __name__ == "__main__":
    main()