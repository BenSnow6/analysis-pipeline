"""
Spectral analysis methods for RPM estimation.

This module implements Welch PSD and STFT methods for extracting
RPM from vibration signals.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Optional, Dict
import logging
from dataclasses import dataclass

try:
    from .tracking import RPMFrame
except ImportError:
    from tracking import RPMFrame

logger = logging.getLogger(__name__)


def welch_psd(data: np.ndarray, fs: float, win_sec: float, 
              overlap: float = 0.5, window: str = 'hann',
              detrend: str = 'linear', max_freq: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch Power Spectral Density estimate.
    
    Args:
        data: Input signal
        fs: Sampling frequency in Hz
        win_sec: Window length in seconds
        overlap: Overlap fraction (0-1)
        window: Window function name
        detrend: Detrending method
        max_freq: Maximum frequency to include in output (Hz)
        
    Returns:
        Tuple of (frequencies, PSD values) limited to [0, max_freq] Hz
    """
    # Calculate window parameters
    nperseg = int(win_sec * fs)
    noverlap = int(overlap * nperseg)
    
    # Compute Welch PSD
    freqs, psd = signal.welch(data, fs=fs, window=window, nperseg=nperseg,
                             noverlap=noverlap, detrend=detrend)
    
    # Limit to max frequency
    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]
    psd = psd[freq_mask]
    
    logger.debug(f"Welch PSD: {len(freqs)} frequency bins, "
                f"resolution={freqs[1]-freqs[0]:.3f} Hz, "
                f"max freq={freqs[-1]:.1f} Hz")
    
    return freqs, psd


def find_peaks_in_psd(freqs: np.ndarray, psd: np.ndarray, 
                     min_freq: float = 10.0, max_freq: float = 50.0,
                     noise_floor_db: float = 3.0, max_peaks: int = 10) -> List[Dict]:
    """
    Find peaks in PSD within frequency range.
    
    Args:
        freqs: Frequency array
        psd: PSD values
        min_freq: Minimum frequency to search (Hz) - corresponds to 600 RPM
        max_freq: Maximum frequency to search (Hz) - corresponds to 3000 RPM
        noise_floor_db: Peaks must be this many dB above noise floor
        max_peaks: Maximum number of peaks to return
        
    Returns:
        List of peak dictionaries with freq, amplitude, index, sorted by amplitude
    """
    # Convert to dB
    psd_db = 10 * np.log10(psd + 1e-12)
    
    # Find frequency range indices
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    freq_indices = np.where(freq_mask)[0]
    
    if len(freq_indices) == 0:
        logger.warning(f"No frequencies in range {min_freq}-{max_freq} Hz")
        return []
    
    # Calculate noise floor as median of PSD in search range
    psd_range = psd_db[freq_mask]
    noise_floor = np.median(psd_range)
    
    # Find peaks that are above noise floor
    peaks, properties = signal.find_peaks(
        psd_range, 
        height=noise_floor + noise_floor_db,
        distance=int(2 * len(psd_range) / (max_freq - min_freq))  # Min 2 Hz separation
    )
    
    # Convert back to full indices
    peak_indices = freq_indices[peaks]
    
    # Create peak list with additional info
    peak_list = []
    for i, idx in enumerate(peak_indices):
        peak_list.append({
            'freq': freqs[idx],
            'amplitude_db': psd_db[idx],
            'amplitude': psd[idx],
            'index': idx,
            'prominence_db': properties.get('prominences', [0]*len(peaks))[i] if 'prominences' in properties else psd_db[idx] - noise_floor,
            'height_above_noise': psd_db[idx] - noise_floor
        })
    
    # Sort by amplitude and limit to max_peaks
    peak_list.sort(key=lambda x: x['amplitude_db'], reverse=True)
    peak_list = peak_list[:max_peaks]
    
    logger.debug(f"Found {len(peak_list)} peaks in {min_freq}-{max_freq} Hz range, "
                f"noise floor={noise_floor:.1f} dB")
    
    return peak_list


def compute_snr(freqs: np.ndarray, psd: np.ndarray, peak_freq: float,
               band_hz: float = 3.0, exclude_hz: float = 0.5) -> float:
    """
    Compute SNR for a peak in the PSD using local band method.
    
    SNR = 10*log10(Ppeak / Pavg), where Pavg is mean PSD in ±band_hz 
    excluding ±exclude_hz around peak.
    
    Args:
        freqs: Frequency array
        psd: PSD values
        peak_freq: Peak frequency in Hz
        band_hz: Width of band for noise calculation (±band_hz around peak)
        exclude_hz: Width of exclusion zone around peak (±exclude_hz)
        
    Returns:
        SNR in dB
    """
    # Find peak index
    peak_idx = np.argmin(np.abs(freqs - peak_freq))
    peak_power = psd[peak_idx]
    
    # Define noise calculation band
    # Include frequencies within ±band_hz but exclude ±exclude_hz around peak
    freq_diff = np.abs(freqs - peak_freq)
    noise_mask = (freq_diff <= band_hz) & (freq_diff > exclude_hz)
    
    # Check if we have enough points for noise estimation
    if np.sum(noise_mask) < 5:
        logger.warning(f"Insufficient points for SNR calculation at {peak_freq:.1f} Hz")
        # Fallback to wider band
        noise_mask = (freq_diff <= 2*band_hz) & (freq_diff > exclude_hz)
    
    # Compute noise floor as mean of band excluding peak
    noise_floor = np.mean(psd[noise_mask])
    
    # Compute SNR
    snr_db = 10 * np.log10(peak_power / noise_floor)
    
    logger.debug(f"SNR at {peak_freq:.1f} Hz: {snr_db:.1f} dB "
                f"(peak power={10*np.log10(peak_power):.1f} dB, "
                f"noise floor={10*np.log10(noise_floor):.1f} dB)")
    
    return snr_db


def extract_harmonics(freqs: np.ndarray, psd: np.ndarray, 
                     fundamental: float, n_harmonics: int = 5,
                     tolerance: float = 0.02) -> Dict[int, float]:
    """
    Extract harmonic amplitudes given fundamental frequency.
    
    Args:
        freqs: Frequency array
        psd: PSD values
        fundamental: Fundamental frequency in Hz
        n_harmonics: Number of harmonics to extract
        tolerance: Relative frequency tolerance (0.02 = 2%)
        
    Returns:
        Dictionary mapping harmonic number to amplitude
    """
    harmonics = {}
    
    for n in range(1, n_harmonics + 1):
        harmonic_freq = n * fundamental
        
        # Find closest frequency bin
        freq_error = np.abs(freqs - harmonic_freq) / harmonic_freq
        if np.min(freq_error) < tolerance:
            idx = np.argmin(freq_error)
            harmonics[n] = psd[idx]
    
    return harmonics


def stft_analysis(data: np.ndarray, fs: float, win_sec: float = 1.0,
                 hop_sec: float = 0.25, window: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform.
    
    Args:
        data: Input signal
        fs: Sampling frequency in Hz
        win_sec: Window length in seconds
        hop_sec: Hop size in seconds
        window: Window function
        
    Returns:
        Tuple of (times, frequencies, STFT magnitude)
    """
    # Calculate parameters
    nperseg = int(win_sec * fs)
    hop_length = int(hop_sec * fs)
    
    # Compute STFT
    f, t, Zxx = signal.stft(data, fs=fs, window=window, nperseg=nperseg,
                           noverlap=nperseg-hop_length)
    
    # Compute magnitude
    stft_mag = np.abs(Zxx)
    
    logger.debug(f"STFT: {len(t)} time frames, {len(f)} frequency bins")
    
    return t, f, stft_mag


def rpm_from_frequency(freq: float) -> float:
    """Convert frequency in Hz to RPM."""
    return freq * 60.0


def frequency_from_rpm(rpm: float) -> float:
    """Convert RPM to frequency in Hz."""
    return rpm / 60.0


def identify_fundamental(peaks: List[Dict], harmonics_check: bool = True) -> Optional[Dict]:
    """
    Identify fundamental frequency from peak list, handling harmonics.
    
    For twin-balance engines, the fundamental might be suppressed and
    the 2nd harmonic dominant. This function checks harmonic relationships.
    
    Args:
        peaks: List of peak dictionaries from find_peaks_in_psd
        harmonics_check: Whether to check for harmonic relationships
        
    Returns:
        Peak dictionary for most likely fundamental, or None if no valid peaks
    """
    if not peaks:
        return None
    
    if not harmonics_check or len(peaks) == 1:
        # Return strongest peak
        return peaks[0]
    
    # Check if peaks have harmonic relationships
    # For each peak, check if other peaks are its harmonics
    candidates = []
    
    for i, candidate in enumerate(peaks[:3]):  # Check top 3 peaks
        fund_freq = candidate['freq']
        harmonic_score = candidate['amplitude_db']
        
        # Check if other peaks are harmonics of this candidate
        for peak in peaks:
            if peak == candidate:
                continue
                
            # Check if peak is a harmonic (2x, 3x, etc.)
            for n in range(2, 6):
                ratio = peak['freq'] / fund_freq
                if abs(ratio - n) < 0.02 * n:  # 2% tolerance
                    # This peak is likely nth harmonic
                    harmonic_score += peak['amplitude_db'] / n
                    break
        
        candidates.append({
            'peak': candidate,
            'harmonic_score': harmonic_score
        })
    
    # Sort by harmonic score
    candidates.sort(key=lambda x: x['harmonic_score'], reverse=True)
    
    # Log the decision
    if len(candidates) > 1:
        logger.debug(f"Fundamental frequency selection: "
                    f"{candidates[0]['peak']['freq']:.1f} Hz "
                    f"(score={candidates[0]['harmonic_score']:.1f})")
    
    return candidates[0]['peak']


def extract_rpm_from_vibration(vibration_magnitude: np.ndarray, 
                              fs: float,
                              config: dict,
                              timestamp: float,
                              sensor_id: str) -> Optional[RPMFrame]:
    """
    Extract RPM from vibration magnitude signal using Welch PSD.
    
    This is the main entry point for WP-2 processing.
    
    Args:
        vibration_magnitude: Vibration magnitude signal |a_body|
        fs: Sampling frequency in Hz
        config: Configuration dictionary with welch parameters
        timestamp: Timestamp for this data segment
        sensor_id: Sensor identifier
        
    Returns:
        RPMFrame with extracted RPM and quality metrics, or None if extraction fails
    """
    # Extract config parameters
    welch_config = config.get('welch', {})
    win_sec = welch_config.get('win_sec', 6.0)
    overlap = welch_config.get('overlap', 0.5)
    
    # Check if we have enough data
    min_samples = int(win_sec * fs)
    if len(vibration_magnitude) < min_samples:
        logger.warning(f"Insufficient data for Welch PSD: {len(vibration_magnitude)} < {min_samples}")
        return None
    
    # Compute Welch PSD
    freqs, psd = welch_psd(
        vibration_magnitude, 
        fs=fs,
        win_sec=win_sec,
        overlap=overlap,
        max_freq=100.0  # Limit to 0-100 Hz (0-6000 RPM)
    )
    
    # Find peaks in expected RPM range (600-3000 RPM = 10-50 Hz)
    peaks = find_peaks_in_psd(
        freqs, psd,
        min_freq=10.0,  # 600 RPM
        max_freq=50.0,  # 3000 RPM
        noise_floor_db=config.get('peak_detection', {}).get('noise_floor_db', 3.0)
    )
    
    if not peaks:
        logger.warning(f"No peaks found in PSD for sensor {sensor_id}")
        return None
    
    # Identify fundamental frequency
    fundamental_peak = identify_fundamental(peaks, harmonics_check=True)
    if not fundamental_peak:
        return None
    
    # Compute SNR
    snr_db = compute_snr(
        freqs, psd, 
        fundamental_peak['freq'],
        band_hz=config.get('snr', {}).get('band_hz', 3.0),
        exclude_hz=config.get('snr', {}).get('exclude_hz', 0.5)
    )
    
    # Extract harmonics for validation
    harmonics = extract_harmonics(
        freqs, psd,
        fundamental_peak['freq'],
        n_harmonics=config.get('peak_detection', {}).get('max_harmonics', 5)
    )
    
    # Convert to RPM
    rpm = rpm_from_frequency(fundamental_peak['freq'])
    
    # Log results
    logger.info(f"RPM extracted: {rpm:.1f} RPM @ {fundamental_peak['freq']:.2f} Hz, "
               f"SNR={snr_db:.1f} dB, sensor={sensor_id}")
    
    # Create RPM frame
    rpm_frame = RPMFrame(
        time=timestamp,
        rpm=rpm,
        snr_db=snr_db,
        sensor_id=sensor_id,
        method='welch'
    )
    
    # Store additional metadata (could be extended in the future)
    rpm_frame.metadata = {
        'frequency_hz': fundamental_peak['freq'],
        'harmonics': harmonics,
        'amplitude_db': fundamental_peak['amplitude_db']
    }
    
    return rpm_frame


def stft_mag(signal: np.ndarray, fs: float, 
             win_sec: float = 1.0, hop_sec: float = 0.25,
             window: str = 'hann', padding: str = 'zero',
             edge_method: str = 'mirror') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute magnitude STFT with explicit edge handling.
    
    This function computes the Short-Time Fourier Transform magnitude
    spectrogram with configurable window parameters and edge handling.
    
    Args:
        signal: Input signal
        fs: Sampling frequency in Hz
        win_sec: Window length in seconds
        hop_sec: Hop size in seconds  
        window: Window function ('hann', 'hamming', 'blackman')
        padding: Padding type for incomplete windows ('zero', 'constant', 'edge')
        edge_method: Edge handling method ('mirror', 'wrap', 'trim')
        
    Returns:
        Tuple of (time_bins, frequencies, magnitude_spectrogram)
        - time_bins: Time centers of each STFT frame
        - frequencies: Frequency bins 
        - magnitude_spectrogram: Magnitude of STFT (freq x time)
    """
    # Convert time parameters to samples
    nperseg = int(win_sec * fs)
    hop_length = int(hop_sec * fs)
    noverlap = nperseg - hop_length
    
    # Handle edge effects based on method
    if edge_method == 'mirror':
        # Pad signal by mirroring edges
        pad_length = nperseg // 2
        signal_padded = np.pad(signal, pad_length, mode='reflect')
        boundary = None  # scipy handles remaining edge effects
    elif edge_method == 'wrap':
        # Pad signal by wrapping around
        pad_length = nperseg // 2
        signal_padded = np.pad(signal, pad_length, mode='wrap')
        boundary = None
    elif edge_method == 'trim':
        # No padding, trim incomplete windows
        signal_padded = signal
        boundary = None
    else:
        # Default: use scipy's boundary handling
        signal_padded = signal
        boundary = padding  # 'zero' or other scipy modes
    
    # Compute STFT
    from scipy import signal as scipy_signal
    freqs, times, stft_complex = scipy_signal.stft(
        signal_padded,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=boundary,
        padded=True if edge_method in ['mirror', 'wrap'] else False
    )
    
    # Compute magnitude
    magnitude = np.abs(stft_complex)
    
    # Adjust time bins to account for padding
    if edge_method in ['mirror', 'wrap']:
        # Shift time bins back to original signal timeline
        time_shift = pad_length / fs
        times = times - time_shift
        
        # Trim outputs to match original signal duration
        valid_time_mask = (times >= 0) & (times <= len(signal) / fs)
        times = times[valid_time_mask]
        magnitude = magnitude[:, valid_time_mask]
    
    logger.debug(
        f"STFT computed: {len(freqs)} freq bins, {len(times)} time bins, "
        f"freq resolution={freqs[1]-freqs[0]:.2f} Hz, "
        f"time resolution={hop_sec:.2f} s"
    )
    
    return times, freqs, magnitude


def extract_rpm_stft(vibration_magnitude: np.ndarray,
                    fs: float,
                    config: dict,
                    start_time: float,
                    sensor_id: str) -> 'RPMTimeSeries':
    """
    Extract time-resolved RPM using STFT with early SNR gating.
    
    This function processes vibration data using STFT to extract RPM
    values at regular time intervals. Low-SNR time bins are immediately
    gated (set to NaN) to provide only confident estimates.
    
    Args:
        vibration_magnitude: Vibration magnitude signal |a_body|
        fs: Sampling frequency in Hz
        config: Configuration dictionary with wp3/stft parameters
        start_time: Start time of the data segment (for time alignment)
        sensor_id: Sensor identifier
        
    Returns:
        RPMTimeSeries with time-resolved RPM estimates
    """
    try:
        from .tracking import RPMTimeSeries, RPMFrame
    except ImportError:
        from tracking import RPMTimeSeries, RPMFrame
    
    # Extract WP-3 config
    wp3_config = config.get('wp3', {})
    stft_config = wp3_config.get('stft', config.get('stft', {}))
    quality_config = wp3_config.get('quality', {})
    
    # STFT parameters
    win_sec = stft_config.get('win_sec', 1.0)
    hop_sec = stft_config.get('hop_sec', 0.25)
    window = stft_config.get('window', 'hann')
    padding = stft_config.get('padding', 'zero')
    edge_method = stft_config.get('edge_method', 'mirror')
    
    # Quality parameters
    min_snr_db = quality_config.get('min_snr_db', 10.0)
    
    # Compute STFT
    times, freqs, magnitude = stft_mag(
        vibration_magnitude,
        fs=fs,
        win_sec=win_sec,
        hop_sec=hop_sec,
        window=window,
        padding=padding,
        edge_method=edge_method
    )
    
    # Prepare storage for results
    rpm_frames = []
    
    # Process each time slice
    for i, t in enumerate(times):
        # Get magnitude spectrum for this time slice
        mag_spectrum = magnitude[:, i]
        
        # Convert to PSD-like values for compatibility with existing peak detection
        psd_slice = mag_spectrum ** 2
        
        # Find peaks using existing function
        peaks = find_peaks_in_psd(
            freqs, psd_slice,
            min_freq=10.0,  # 600 RPM
            max_freq=50.0,  # 3000 RPM
            noise_floor_db=config.get('peak_detection', {}).get('noise_floor_db', 3.0)
        )
        
        if not peaks:
            # No peaks found - create invalid frame
            rpm_frame = RPMFrame(
                time=start_time + t,
                rpm=np.nan,
                snr_db=0.0,
                sensor_id=sensor_id,
                method='stft',
                metadata={'valid': False, 'reason': 'no_peaks'}
            )
        else:
            # Identify fundamental frequency
            fundamental_peak = identify_fundamental(
                peaks, 
                harmonics_check=True
            )
            
            # Calculate SNR for the peak
            snr_db = compute_snr(
                freqs, psd_slice,
                fundamental_peak['freq'],
                band_hz=config.get('snr', {}).get('band_hz', 3.0),
                exclude_hz=config.get('snr', {}).get('exclude_hz', 0.5)
            )
            
            # Early SNR gating
            if snr_db < min_snr_db:
                # Low SNR - gate this estimate
                rpm_frame = RPMFrame(
                    time=start_time + t,
                    rpm=np.nan,
                    snr_db=snr_db,
                    sensor_id=sensor_id,
                    method='stft',
                    metadata={'valid': False, 'reason': 'low_snr', 
                             'detected_freq': fundamental_peak['freq']}
                )
            else:
                # Valid estimate
                rpm = fundamental_peak['freq'] * 60.0
                rpm_frame = RPMFrame(
                    time=start_time + t,
                    rpm=rpm,
                    snr_db=snr_db,
                    sensor_id=sensor_id,
                    method='stft',
                    metadata={'valid': True, 
                             'frequency_hz': fundamental_peak['freq'],
                             'amplitude_db': fundamental_peak['amplitude_db']}
                )
        
        rpm_frames.append(rpm_frame)
    
    # Create time series
    rpm_series = RPMTimeSeries(
        frames=rpm_frames,
        metadata={
            'sensor_id': sensor_id,
            'method': 'stft',
            'stft_params': {
                'win_sec': win_sec,
                'hop_sec': hop_sec,
                'window': window,
                'edge_method': edge_method
            },
            'snr_threshold': min_snr_db,
            'start_time': start_time,
            'duration': len(vibration_magnitude) / fs
        }
    )
    
    # Log summary
    valid_count = sum(1 for f in rpm_frames if not np.isnan(f.rpm))
    total_count = len(rpm_frames)
    availability = 100.0 * valid_count / total_count if total_count > 0 else 0.0
    
    logger.info(
        f"STFT RPM extraction complete",
        sensor=sensor_id,
        valid_frames=valid_count,
        total_frames=total_count,
        availability_pct=round(availability, 1)
    )
    
    return rpm_series