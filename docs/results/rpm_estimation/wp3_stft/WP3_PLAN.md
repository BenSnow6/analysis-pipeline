# WP-3: STFT + Order Tracking for Transients - Implementation Plan

## Overview

Work Package 3 extends the RPM estimation system to handle transient conditions where engine speed changes rapidly. Using Short-Time Fourier Transform (STFT), we achieve 4 Hz temporal resolution while maintaining frequency accuracy. This package includes robust quality controls, proper edge handling, and lightweight smoothing for high-rate RPM changes.

## Core Requirements

### From Vibration Plan
- STFT implementation with 1-second windows, 0.25s hop
- Time-resolved RPM extraction
- Optional Vold-Kalman order tracking for ΔRPM/Δt > 150 RPM/s
- HDF5 output format compatible with WP-2

### Enhanced Requirements (Polish Points)
1. **Early SNR gating**: Drop low-confidence time bins immediately
2. **Anti-alias verification**: Ensure 80-90 Hz filter was applied
3. **Explicit edge handling**: Document padding scheme for first/last windows
4. **Triangular ramp testing**: Validate bidirectional RPM changes
5. **Lightweight smoothing**: Simple methods before heavy order tracking

## Implementation Components

### 1. Pre-Processing Validation (`quality.py` extension)
```python
def verify_antialiasing_filter(qa_summary: dict) -> bool:
    """
    Verify that anti-aliasing filter was applied in WP-1.
    
    Args:
        qa_summary: Quality assessment summary from WP-1
        
    Returns:
        True if filter verified, False otherwise
    """
```

### 2. STFT Core Implementation (`spectral.py`)
```python
def stft_mag(signal: np.ndarray, fs: float, 
             win_sec: float = 1.0, hop_sec: float = 0.25,
             window: str = 'hann', padding: str = 'zero',
             edge_method: str = 'mirror') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute magnitude STFT with explicit edge handling.
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        win_sec: Window length in seconds
        hop_sec: Hop size in seconds
        window: Window function
        padding: Padding type ('zero', 'constant', 'edge')
        edge_method: Edge handling ('mirror', 'wrap', 'trim')
        
    Returns:
        (time_bins, frequencies, magnitude_spectrogram)
    """
```

### 3. Time-Resolved RPM Extraction (`spectral.py`)
```python
def extract_rpm_stft(signal: np.ndarray, fs: float, config: dict,
                     start_time: float, sensor_id: str) -> RPMTimeSeries:
    """
    Extract time-resolved RPM using STFT with early SNR gating.
    
    For each time slice:
    - Apply peak detection
    - Calculate SNR immediately
    - Gate low-SNR bins (return NaN)
    - Build sparse but confident RPM series
    """
```

### 4. Smoothing Module (`tracking.py`)
```python
def smooth_rpm_series(time: np.ndarray, rpm: np.ndarray, 
                     method: str = 'polynomial', 
                     window: int = 5) -> np.ndarray:
    """
    Apply lightweight smoothing to RPM series.
    
    Methods:
    - polynomial: Fit low-order polynomial in sliding window
    - median: Median filter with outlier rejection
    - moving_avg: Weighted moving average
    
    Only applied to high-rate regions (>150 RPM/s).
    """
```

### 5. Main Processing Script (`wp3_process.py`)
```python
def process_experiment(experiment: str, session: str, sensors: List[str],
                      config: dict) -> Dict[str, RPMTimeSeries]:
    """
    Process one experiment with STFT-based RPM extraction.
    
    Steps:
    1. Load WP-1 processed data
    2. Verify anti-aliasing filter
    3. Apply STFT with proper edge handling
    4. Extract RPM with SNR gating
    5. Optional smoothing for transients
    6. Save to HDF5
    """
```

## Test Suite Additions

### Unit Tests (`tests/test_stft.py`)
1. **Basic STFT**: Verify against scipy reference
2. **Edge effects**: Check first/last window handling
3. **SNR gating**: Confirm low-SNR rejection
4. **Triangular ramp**: 500→2000→500 RPM over 10s
5. **Anti-alias check**: Test filter verification logic

### Integration Tests
1. Compare with WP-2 on steady segments (< 5 RPM difference)
2. Process 026_Engine_rpm_sweep with known transitions
3. Verify exact time alignment with raw data

## Output Format

### HDF5 Structure
```
experiment_sensor_stft.h5
├── /metadata/
│   ├── experiment: str
│   ├── session: str
│   ├── sensor: str
│   ├── method: 'stft'
│   ├── anti_alias_verified: bool
│   ├── edge_padding_method: str
│   ├── processing_timestamp: str
│   └── config: dict (serialized)
├── /data/
│   ├── time: float[N] (exact experiment time)
│   ├── rpm_est: float[N] (NaN for gated bins)
│   ├── snr_db: float[N]
│   ├── confidence: float[N] (0-1 normalized)
│   ├── valid: bool[N]
│   └── smoothed_rpm: float[N] (if applied)
└── /quality/
    ├── availability: float (% valid bins)
    ├── mean_snr: float
    └── max_delta_rpm: float

```

## Configuration Updates

### rpm_config.yaml additions
```yaml
wp3:
  # STFT parameters
  stft:
    win_sec: 1.0
    hop_sec: 0.25
    window: 'hann'
    padding: 'zero'
    edge_method: 'mirror'
    
  # Quality control
  quality:
    min_snr_db: 10.0  # Early gating threshold
    require_antialiasing: true
    
  # Smoothing
  smoothing:
    enabled: true
    method: 'polynomial'  # polynomial, median, moving_avg
    window_size: 5
    high_rate_threshold: 150  # RPM/s
    
  # Output
  output:
    format: 'hdf5'
    include_smoothed: true
    sparse_output: true  # Only save valid bins
```

## CLI Integration

### New command options
```bash
# Basic WP-3 processing
python -m rpm_estimation.cli --wp 3 --exp 026_Engine_rpm_sweep --session afternoon

# With custom SNR threshold
python -m rpm_estimation.cli --wp 3 --exp 016_Straight_cruise_1 \
    --session afternoon --snr-threshold 8.0

# Disable smoothing
python -m rpm_estimation.cli --wp 3 --exp 007_Fast_stbd_turn_1 \
    --session afternoon --no-smoothing

# Custom edge padding
python -m rpm_estimation.cli --wp 3 --exp 009_Fast_port_turn_1 \
    --session afternoon --edge-padding mirror
```

## Implementation Timeline

| Task | Time | Priority | Dependencies |
|------|------|----------|--------------|
| Anti-alias verification | 30 min | High | WP-1 QA files |
| STFT core + edge handling | 2 hrs | High | - |
| SNR gating integration | 1 hr | High | STFT core |
| Triangular ramp test | 1 hr | High | STFT core |
| Lightweight smoother | 1.5 hrs | Medium | RPM extraction |
| Full pipeline | 2 hrs | High | All above |
| CLI updates | 30 min | Medium | Pipeline |
| Documentation | 1 hr | Medium | All complete |

Total: ~9 hours

## Success Metrics

1. **Functionality**
   - All unit tests pass
   - Triangular ramp RMSE < 20 RPM
   - Edge effects properly handled

2. **Quality**
   - SNR gating reduces false positives to < 5%
   - Anti-alias check prevents polluted estimates
   - Time axis aligns exactly with raw data

3. **Performance**
   - Processing time < 1 minute per experiment
   - Memory usage < 500 MB per sensor
   - HDF5 file size < 10 MB per experiment

4. **Compatibility**
   - Seamless integration with WP-2 outputs
   - Ready for WP-4 multi-sensor fusion
   - CLI maintains consistent interface

## Risk Mitigation

1. **Edge effects**: Explicit padding documentation, validation tests
2. **Low SNR data**: Early gating prevents propagation
3. **Aliasing**: Mandatory filter check with abort option
4. **Over-smoothing**: Only apply to high-rate regions
5. **Time alignment**: Careful indexing with validation

## Next Steps After WP-3

- WP-4: Multi-sensor fusion using confident estimates
- WP-5: Validation against ground truth
- WP-6: Batch processing all experiments

This plan incorporates all feedback while maintaining compatibility with the existing system.