"""
Spectral analysis methods for RPM estimation.

This module implements Welch PSD and STFT methods for extracting
RPM from vibration signals.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def welch_psd(data: np.ndarray, fs: float, win_sec: float, 
              overlap: float = 0.5, window: str = 'hann',
              detrend: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch Power Spectral Density estimate.
    
    Args:
        data: Input signal
        fs: Sampling frequency in Hz
        win_sec: Window length in seconds
        overlap: Overlap fraction (0-1)
        window: Window function name
        detrend: Detrending method
        
    Returns:
        Tuple of (frequencies, PSD values)
    """
    # Calculate window parameters
    nperseg = int(win_sec * fs)
    noverlap = int(overlap * nperseg)
    
    # Compute Welch PSD
    freqs, psd = signal.welch(data, fs=fs, window=window, nperseg=nperseg,
                             noverlap=noverlap, detrend=detrend)
    
    logger.debug(f"Welch PSD: {len(freqs)} frequency bins, "
                f"resolution={freqs[1]-freqs[0]:.3f} Hz")
    
    return freqs, psd


def find_peaks_in_psd(freqs: np.ndarray, psd: np.ndarray, 
                     min_freq: float = 10.0, max_freq: float = 50.0,
                     prominence_db: float = 3.0) -> List[Dict]:
    """
    Find peaks in PSD within frequency range.
    
    Args:
        freqs: Frequency array
        psd: PSD values
        min_freq: Minimum frequency to search (Hz)
        max_freq: Maximum frequency to search (Hz)
        prominence_db: Minimum peak prominence in dB
        
    Returns:
        List of peak dictionaries with freq, amplitude, index
    """
    # Convert to dB
    psd_db = 10 * np.log10(psd + 1e-12)
    
    # Find frequency range indices
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    freq_indices = np.where(freq_mask)[0]
    
    if len(freq_indices) == 0:
        return []
    
    # Find peaks in the frequency range
    psd_range = psd_db[freq_mask]
    peaks, properties = signal.find_peaks(psd_range, prominence=prominence_db)
    
    # Convert back to full indices
    peak_indices = freq_indices[peaks]
    
    # Create peak list
    peak_list = []
    for i, idx in enumerate(peak_indices):
        peak_list.append({
            'freq': freqs[idx],
            'amplitude_db': psd_db[idx],
            'amplitude': psd[idx],
            'index': idx,
            'prominence_db': properties['prominences'][i]
        })
    
    # Sort by amplitude
    peak_list.sort(key=lambda x: x['amplitude_db'], reverse=True)
    
    logger.debug(f"Found {len(peak_list)} peaks in {min_freq}-{max_freq} Hz range")
    
    return peak_list


def compute_snr(freqs: np.ndarray, psd: np.ndarray, peak_freq: float,
               band_width: float = 3.0) -> float:
    """
    Compute SNR for a peak in the PSD.
    
    Args:
        freqs: Frequency array
        psd: PSD values
        peak_freq: Peak frequency in Hz
        band_width: Width of exclusion band around peak (Hz)
        
    Returns:
        SNR in dB
    """
    # Find peak index
    peak_idx = np.argmin(np.abs(freqs - peak_freq))
    peak_power = psd[peak_idx]
    
    # Define noise band (exclude ±band_width/2 around peak)
    noise_mask = np.abs(freqs - peak_freq) > band_width / 2
    
    # Compute noise floor as mean of surrounding spectrum
    noise_floor = np.mean(psd[noise_mask])
    
    # Compute SNR
    snr_db = 10 * np.log10(peak_power / noise_floor)
    
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