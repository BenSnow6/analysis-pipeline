# WP-3: STFT + Order Tracking for Transients

## Overview

Work Package 3 extends the RPM estimation system to handle transient conditions using Short-Time Fourier Transform (STFT). This implementation provides:

- **4 Hz temporal resolution** for tracking RPM changes
- **Early SNR gating** to ensure only confident estimates
- **Explicit edge handling** for accurate time alignment
- **Lightweight smoothing** for high-rate RPM changes
- **Anti-aliasing verification** to prevent spectral pollution

## Key Features

### 1. Robust STFT Implementation
- 1-second windows with 0.25s hop (75% overlap)
- Configurable edge handling (mirror, wrap, trim)
- Exact time alignment with original data

### 2. Quality Control
- Early SNR gating at slice level (default: 10 dB)
- Anti-aliasing filter verification from WP-1
- Sparse output with NaN for low-confidence bins

### 3. Adaptive Smoothing
- Automatic detection of high-rate regions (>150 RPM/s)
- Multiple smoothing methods (polynomial, median, moving average)
- Preserves steady-state accuracy

### 4. Comprehensive Testing
- Unit tests with synthetic signals
- Triangular ramp test (500→2000→500 RPM)
- Edge effect validation

## Installation

WP-3 uses the same environment as WP-1 and WP-2:

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Dependencies are already installed via requirements.txt
```

## Usage

### Command Line Interface

Basic usage:
```bash
python -m rpm_estimation.cli --wp 3 --exp 026_Engine_rpm_sweep --session afternoon
```

With custom parameters:
```bash
python -m rpm_estimation.cli --wp 3 --exp 016_Straight_cruise_1 \
    --session afternoon \
    --sensors Sensor_3 Sensor_wb \
    --snr-threshold 8.0 \
    --plot
```

Disable smoothing:
```bash
python -m rpm_estimation.cli --wp 3 --exp 007_Fast_stbd_turn_1 \
    --session afternoon \
    --no-smoothing
```

Batch processing:
```bash
python -m rpm_estimation.cli --wp 3 --all --session afternoon
```

### Direct Script Usage

```bash
python wp3_process.py --experiment 026_Engine_rpm_sweep --session afternoon
```

### Python API

```python
from rpm_estimation.wp3_process import process_experiment

# Process single experiment
results = process_experiment(
    experiment='026_Engine_rpm_sweep',
    session='afternoon',
    sensors=['Sensor_3', 'Sensor_4'],
    generate_plots=True
)

# Access results
for sensor_id, output_path in results.items():
    print(f"{sensor_id}: {output_path}")
```

## Configuration

Key parameters in `rpm_config.yaml`:

```yaml
wp3:
  # STFT parameters
  stft:
    win_sec: 1.0        # Window length
    hop_sec: 0.25       # Hop size (4 Hz update)
    edge_method: 'mirror'  # Edge handling
    
  # Quality control
  quality:
    min_snr_db: 10.0    # SNR threshold
    require_antialiasing: true
    
  # Smoothing
  smoothing:
    enabled: true
    method: 'polynomial'
    high_rate_threshold: 150  # RPM/s
```

## Output Format

### HDF5 Structure
```
results/wp3/<session>/<experiment>_<sensor>_stft.h5
├── /metadata/
│   ├── experiment, session, sensor
│   ├── anti_alias_verified
│   ├── stft_parameters
│   └── processing_timestamp
├── /data/
│   ├── time         # Exact alignment with experiment
│   ├── rpm_est      # RPM values (NaN for gated)
│   ├── snr_db       # SNR for each time bin
│   ├── valid        # Boolean validity flags
│   └── smoothed_rpm # If smoothing enabled
└── /quality/
    ├── availability  # % valid estimates
    ├── mean_snr
    └── max_delta_rpm
```

### Diagnostic Plots
Three-panel plots showing:
1. RPM over time with valid/gated points
2. SNR over time with threshold line
3. RPM rate of change

Located in: `results/wp3/plots/<session>/<experiment>_<sensor>_stft_diagnostic.png`

## Algorithm Details

### STFT Processing
1. Apply configurable edge padding
2. Compute STFT with scipy.signal.stft
3. Extract magnitude spectrogram
4. Adjust time bins for exact alignment

### RPM Extraction per Time Slice
1. Convert magnitude to PSD-like values
2. Find peaks using WP-2 peak detection
3. Identify fundamental frequency
4. Calculate SNR using local band method
5. Gate if SNR < threshold

### Smoothing (High-Rate Regions Only)
1. Calculate RPM rate of change
2. Identify regions > 150 RPM/s
3. Apply selected smoothing method
4. Preserve steady-state regions

## Validation

### Run Unit Tests
```bash
pytest tests/test_stft.py -v
```

### Key Test Cases
- **Basic STFT**: Frequency/time resolution
- **Edge Effects**: Different padding methods
- **SNR Gating**: Low/high SNR behavior
- **Triangular Ramp**: 500→2000→500 RPM tracking
- **Anti-alias Check**: Filter verification

## Troubleshooting

### Common Issues

1. **"Anti-aliasing filter verification failed"**
   - Check WP-1 processing included filtering
   - Verify qa_summary.json exists
   - Use `--no-antialiasing-check` to bypass (not recommended)

2. **Low availability (<50%)**
   - Reduce SNR threshold: `--snr-threshold 7.0`
   - Check data quality with WP-1 reports
   - Verify correct sensor selection

3. **Over-smoothing**
   - Disable smoothing: `--no-smoothing`
   - Adjust high-rate threshold in config
   - Try different smoothing method

4. **Memory issues with large files**
   - Process sensors individually
   - Reduce STFT window overlap
   - Enable sparse output mode

### Debug Mode
```bash
python -m rpm_estimation.cli --wp 3 --exp test --session morning \
    --log-level DEBUG --dry-run
```

## Performance

Typical processing times (per experiment):
- Single sensor: 10-30 seconds
- All sensors: 30-90 seconds
- With plots: +10-20 seconds

Memory usage:
- ~200 MB per sensor
- Scales with experiment duration

## Comparison with WP-2

| Feature | WP-2 (Welch) | WP-3 (STFT) |
|---------|--------------|-------------|
| Time Resolution | 30s windows | 0.25s (4 Hz) |
| Frequency Resolution | 0.167 Hz | 1 Hz |
| Best For | Steady-state | Transients |
| SNR Performance | Better | Good |
| Output Size | Small | Larger |

## Next Steps

After successful WP-3 processing:

1. **WP-4**: Multi-sensor fusion using confident estimates
2. **WP-5**: Validation against ground truth
3. **WP-6**: Batch processing all experiments

## References

1. Vibration plan: See `../../../vibration_plan.md`
2. WP-3 implementation: See `WP3_PLAN.md`
3. Test results: See test output in `results/wp3/`

## Changelog

- 2025-06-20: Initial implementation
  - Core STFT with edge handling
  - Early SNR gating
  - Lightweight smoothing
  - Anti-alias verification
  - Triangular ramp test