"""
Signal preprocessing for RPM estimation.

This module implements filtering, detrending, and signal conditioning
operations for vibration data.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import time
from .logging_config import get_logger, ProcessingError, log_processing_start, log_quality_summary
from .io import load_aligned_data, save_processed_data
from .quality import assess_signal_quality, generate_quality_report, save_quality_report, check_multi_axis_quality
from .schema import create_parquet_metadata

logger = get_logger("preprocess")


def high_pass_filter(data: np.ndarray, fs: float, cutoff: float, 
                    order: int = 4) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to remove low-frequency content.
    
    Args:
        data: Input signal
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz
        order: Filter order (default: 4)
        
    Returns:
        Filtered signal
    """
    # Design Butterworth high-pass filter
    sos = signal.butter(order, cutoff, btype='highpass', fs=fs, output='sos')
    
    # Apply filter (using filtfilt for zero phase)
    filtered = signal.sosfiltfilt(sos, data)
    
    logger.debug(f"Applied HP filter: cutoff={cutoff}Hz, order={order}")
    return filtered


def compute_vibration_magnitude(accel_x: np.ndarray, accel_y: np.ndarray, 
                              accel_z: np.ndarray) -> np.ndarray:
    """
    Compute vibration magnitude from 3-axis accelerometer data.
    
    Args:
        accel_x: X-axis acceleration
        accel_y: Y-axis acceleration
        accel_z: Z-axis acceleration
        
    Returns:
        Vibration magnitude |a|
    """
    magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    return magnitude


def remove_gravity(accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray,
                  gravity: float = 9.80665) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove gravity component from accelerometer data.
    
    Args:
        accel_x: X-axis acceleration
        accel_y: Y-axis acceleration 
        accel_z: Z-axis acceleration
        gravity: Gravity constant (default: 9.80665 m/s²)
        
    Returns:
        Tuple of gravity-removed accelerations (x, y, z)
    """
    # Estimate static gravity vector (mean over signal)
    gx_mean = np.mean(accel_x)
    gy_mean = np.mean(accel_y)
    gz_mean = np.mean(accel_z)
    
    # Remove mean (gravity) component
    accel_x_nograv = accel_x - gx_mean
    accel_y_nograv = accel_y - gy_mean
    accel_z_nograv = accel_z - gz_mean
    
    logger.debug(f"Removed gravity: [{gx_mean:.3f}, {gy_mean:.3f}, {gz_mean:.3f}] m/s²")
    
    return accel_x_nograv, accel_y_nograv, accel_z_nograv


def apply_anti_alias_filter(data: np.ndarray, fs: float, cutoff: float,
                          order: int = 4) -> np.ndarray:
    """
    Apply anti-aliasing low-pass filter.
    
    Args:
        data: Input signal
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz
        order: Filter order (default: 4)
        
    Returns:
        Filtered signal
    """
    # Design Butterworth low-pass filter
    sos = signal.butter(order, cutoff, btype='lowpass', fs=fs, output='sos')
    
    # Apply filter
    filtered = signal.sosfiltfilt(sos, data)
    
    logger.debug(f"Applied anti-alias filter: cutoff={cutoff}Hz, order={order}")
    return filtered


def compute_quality_metrics(data: np.ndarray, window_size: int = None) -> dict:
    """
    Compute signal quality metrics.
    
    Args:
        data: Input signal
        window_size: Window size for rolling metrics (samples)
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        'rms': np.sqrt(np.mean(data**2)),
        'peak': np.max(np.abs(data)),
        'mean': np.mean(data),
        'std': np.std(data),
        'kurtosis': stats.kurtosis(data),
        'peak_to_rms': np.max(np.abs(data)) / np.sqrt(np.mean(data**2))
    }
    
    # Check for clipping
    max_val = np.max(np.abs(data))
    if max_val > 0:
        clipping_ratio = np.sum(np.abs(data) > 0.95 * max_val) / len(data)
        metrics['clipping_ratio'] = clipping_ratio
        metrics['is_clipped'] = clipping_ratio > 0.01  # More than 1% at max
    
    return metrics


def detrend_signal(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Remove trend from signal.
    
    Args:
        data: Input signal
        method: Detrending method ('linear', 'constant')
        
    Returns:
        Detrended signal
    """
    if method == 'linear':
        detrended = signal.detrend(data, type='linear')
    elif method == 'constant':
        detrended = signal.detrend(data, type='constant')
    else:
        raise ValueError(f"Unknown detrend method: {method}")
    
    return detrended


def preprocess_for_rpm(accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray,
                      fs: float, config: dict) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline for RPM estimation.
    
    Args:
        accel_x: X-axis acceleration
        accel_y: Y-axis acceleration
        accel_z: Z-axis acceleration
        fs: Sampling frequency
        config: Configuration dictionary
        
    Returns:
        Tuple of (processed magnitude signal, quality metrics)
    """
    # Remove gravity
    ax_nograv, ay_nograv, az_nograv = remove_gravity(accel_x, accel_y, accel_z)
    
    # Compute magnitude
    magnitude = compute_vibration_magnitude(ax_nograv, ay_nograv, az_nograv)
    
    # High-pass filter
    hp_cutoff = config.get('hp_cutoff', 5.0)
    magnitude_hp = high_pass_filter(magnitude, fs, hp_cutoff)
    
    # Detrend
    magnitude_detrend = detrend_signal(magnitude_hp, method='linear')
    
    # Compute quality metrics
    metrics = compute_quality_metrics(magnitude_detrend)
    
    logger.info(f"Preprocessing complete: RMS={metrics['rms']:.3f}, "
                f"Peak/RMS={metrics['peak_to_rms']:.2f}")
    
    return magnitude_detrend, metrics


def process_sensor_wp1(experiment: str, session: str, sensor_id: str, 
                      config: Dict[str, Any], 
                      base_path: Optional[Path] = None,
                      output_base: Optional[Path] = None) -> Dict[str, Any]:
    """
    Process a single sensor through the complete WP-1 pipeline.
    
    Args:
        experiment: Experiment name
        session: Session (morning/afternoon)
        sensor_id: Sensor identifier
        config: Configuration dictionary
        base_path: Base path for input data
        output_base: Base path for output data
        
    Returns:
        Dictionary with processing results and status
    """
    start_time = time.time()
    
    # Set up logger context
    logger.set_context(experiment=experiment, session=session, sensor=sensor_id)
    
    try:
        # Step 1: Load aligned data with rotation
        logger.info("Loading aligned data", processing_step="data_loading")
        df = load_aligned_data(experiment, session, sensor_id, base_path, apply_rotation=True)
        
        # Validate we have required columns
        required_cols = ['time_from_sync', 't', 'x_body', 'y_body', 'z_body']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Step 2: Extract configuration parameters
        wp1_config = config.get('wp1', {})
        filters_config = wp1_config.get('filters', {})
        processing_config = wp1_config.get('processing', {})
        
        # Step 3: Apply preprocessing
        logger.info("Applying preprocessing", processing_step="preprocessing")
        
        # Extract body-frame accelerations
        accel_x = df['x_body'].values
        accel_y = df['y_body'].values
        accel_z = df['z_body'].values
        
        # Remove gravity if configured
        if processing_config.get('remove_gravity', True):
            accel_x, accel_y, accel_z = remove_gravity(accel_x, accel_y, accel_z)
        
        # Apply high-pass filter
        hp_cutoff = filters_config.get('highpass_cutoff', 5.0)
        hp_order = filters_config.get('highpass_order', 4)
        
        accel_x_hp = high_pass_filter(accel_x, config['fs'], hp_cutoff, hp_order)
        accel_y_hp = high_pass_filter(accel_y, config['fs'], hp_cutoff, hp_order)
        accel_z_hp = high_pass_filter(accel_z, config['fs'], hp_cutoff, hp_order)
        
        # Detrend if configured
        detrend_method = processing_config.get('detrend_method', 'linear')
        if detrend_method:
            accel_x_hp = detrend_signal(accel_x_hp, method=detrend_method)
            accel_y_hp = detrend_signal(accel_y_hp, method=detrend_method)
            accel_z_hp = detrend_signal(accel_z_hp, method=detrend_method)
        
        # Compute magnitude
        accel_mag_hp = compute_vibration_magnitude(accel_x_hp, accel_y_hp, accel_z_hp)
        
        # Step 4: Quality assessment
        logger.info("Performing quality assessment", processing_step="quality_assessment")
        
        # Assess signal quality
        quality_results = assess_signal_quality(
            accel_mag_hp, 
            df['time_from_sync'].values,
            config,
            sensor_id
        )
        
        # Check per-axis quality
        axes_quality = check_multi_axis_quality(accel_x_hp, accel_y_hp, accel_z_hp, config)
        
        # Log quality summary
        log_quality_summary(
            logger,
            sensor_id,
            quality_results['summary']['total_windows'],
            quality_results['summary']['clipped_windows']
        )
        
        # Step 5: Create output DataFrame
        output_df = df.copy()
        
        # Add processed columns
        output_df['a_hp_x'] = accel_x_hp
        output_df['a_hp_y'] = accel_y_hp
        output_df['a_hp_z'] = accel_z_hp
        output_df['a_hp_mag'] = accel_mag_hp
        
        # Add quality flags
        quality_flag = np.zeros(len(df), dtype=np.int8)
        
        # Map window results to samples
        for window in quality_results['windows']:
            start_idx = window['window_id'] * int(config['fs'] * quality_results['parameters_used']['window_sec'])
            end_idx = min(start_idx + int(config['fs'] * quality_results['parameters_used']['window_sec']), len(df))
            
            if window['metrics']['clipping_detected']:
                quality_flag[start_idx:end_idx] = 2  # Bad
            elif window['metrics']['peak_to_rms'] > 8:
                quality_flag[start_idx:end_idx] = 1  # Warning
            
            # Set window ID
            if 'window_id' not in output_df.columns:
                output_df['window_id'] = -1
            output_df.loc[start_idx:end_idx-1, 'window_id'] = window['window_id']
        
        output_df['quality_flag'] = quality_flag
        
        # Step 6: Save outputs
        if output_base is None:
            output_base = Path(__file__).parent.parent.parent / 'aligned_data'
        
        output_dir = output_base / session / experiment
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Parquet file
        parquet_path = output_dir / f"proc_IMU_{sensor_id}.parquet"
        
        # Create metadata
        metadata = create_parquet_metadata(experiment, session, sensor_id, config)
        
        # Save with metadata
        save_processed_data(
            output_df,
            parquet_path,
            compression=wp1_config.get('output', {}).get('parquet_compression', 'snappy'),
            schema_version=wp1_config.get('output', {}).get('schema_version', '1.0')
        )
        
        # Generate and save quality report
        quality_report = generate_quality_report(
            quality_results,
            experiment,
            session,
            config_version=wp1_config.get('output', {}).get('schema_version', '1.0')
        )
        
        # Add axes quality to report
        quality_report['axes_quality'] = axes_quality
        
        # Save quality report
        quality_path = output_dir / f"qa_summary_{sensor_id}.json"
        save_quality_report(quality_report, quality_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.log_timing("Sensor processing", processing_time * 1000, sample_count=len(df))
        
        # Clear logger context
        logger.clear_context()
        
        return {
            'status': 'success',
            'sensor_id': sensor_id,
            'samples_processed': len(df),
            'quality': quality_results['summary']['overall_quality'],
            'output_files': {
                'parquet': str(parquet_path),
                'quality_report': str(quality_path)
            },
            'processing_time_seconds': processing_time
        }
        
    except Exception as e:
        logger.error(
            f"Failed to process sensor {sensor_id}: {str(e)}",
            error_type=ProcessingError.FATAL,
            processing_step="pipeline_error"
        )
        logger.clear_context()
        
        return {
            'status': 'error',
            'sensor_id': sensor_id,
            'error': str(e),
            'error_type': type(e).__name__
        }


def process_experiment_wp1(experiment: str, session: str, config: Dict[str, Any],
                         sensors: Optional[List[str]] = None,
                         base_path: Optional[Path] = None,
                         output_base: Optional[Path] = None,
                         parallel: bool = True) -> Dict[str, Any]:
    """
    Process all sensors for an experiment through WP-1 pipeline.
    
    Args:
        experiment: Experiment name
        session: Session (morning/afternoon)
        config: Configuration dictionary
        sensors: List of sensors to process (defaults to config)
        base_path: Base path for input data
        output_base: Base path for output data
        parallel: Whether to process sensors in parallel
        
    Returns:
        Dictionary with overall results
    """
    # Set up logging
    log_processing_start(logger, experiment, session)
    
    # Get sensor list
    if sensors is None:
        sensors = config.get('wp1', {}).get('sensors', {}).get('default', 
                           ['Sensor_3', 'Sensor_4', 'Sensor_wb'])
    
    logger.info(f"Processing {len(sensors)} sensors: {sensors}")
    
    results = {}
    
    if parallel and len(sensors) > 1:
        # Parallel processing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=min(len(sensors), 4)) as executor:
            futures = {
                executor.submit(process_sensor_wp1, experiment, session, sensor, 
                              config, base_path, output_base): sensor
                for sensor in sensors
            }
            
            for future in as_completed(futures):
                sensor = futures[future]
                try:
                    results[sensor] = future.result()
                except Exception as e:
                    logger.error(
                        f"Parallel processing failed for {sensor}: {str(e)}",
                        sensor=sensor,
                        error_type=ProcessingError.RECOVERABLE
                    )
                    results[sensor] = {
                        'status': 'error',
                        'sensor_id': sensor,
                        'error': str(e)
                    }
    else:
        # Sequential processing
        for sensor in sensors:
            results[sensor] = process_sensor_wp1(
                experiment, session, sensor, config, base_path, output_base
            )
    
    # Generate summary
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = len(results) - successful
    
    summary = {
        'experiment': experiment,
        'session': session,
        'sensors_processed': len(sensors),
        'successful': successful,
        'failed': failed,
        'sensor_results': results
    }
    
    # Log summary
    logger.info(
        f"Experiment processing complete: {successful}/{len(sensors)} sensors successful",
        successful=successful,
        failed=failed,
        processing_step="completion"
    )
    
    return summary