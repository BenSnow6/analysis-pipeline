# Work Package 2 (WP-2) Implementation

## Overview

WP-2 implements the Welch Power Spectral Density (PSD) analysis for extracting engine RPM from vibration data. This work package:

1. Loads processed vibration data from WP-1
2. Applies Welch PSD with optimized parameters
3. Detects peaks using intelligent algorithms
4. Handles harmonic disambiguation 
5. Calculates SNR for quality assessment
6. Outputs time-series RPM estimates with confidence metrics

## Key Features

### 1. Robust Spectral Analysis
- Welch PSD with 6-second windows and 50% overlap
- Frequency resolution of ~0.167 Hz (10 RPM)
- Limited to 0-100 Hz range (0-6000 RPM)
- Linear detrending for each window

### 2. Intelligent Peak Detection
- Noise floor estimation using median PSD
- Peaks must exceed noise floor by configurable threshold (3 dB default)
- Minimum 2 Hz separation between peaks
- Harmonic relationship checking

### 3. SNR Calculation
- Local band method: ±3 Hz around peak
- Excludes ±0.5 Hz around peak center
- Fallback to wider band if insufficient points

### 4. Harmonic Disambiguation
- Identifies fundamental frequency even when 2nd harmonic is stronger
- Harmonic scoring algorithm considers multiple peaks
- Handles twin-balance engine characteristics

## Algorithm Details

### Welch PSD Parameters
```yaml
welch:
  win_sec: 6.0      # 6-second windows
  overlap: 0.5      # 50% overlap
  window: 'hann'    # Hann window
  detrend: 'linear' # Remove linear trends
```

### Peak Detection
1. Convert PSD to dB scale
2. Calculate noise floor as median in search range
3. Find peaks above noise floor + threshold
4. Sort by amplitude and limit to top N peaks

### Fundamental Identification
1. For each candidate peak, check if others are its harmonics
2. Score based on amplitude + harmonic relationships
3. Select candidate with highest harmonic score

### SNR Calculation
```
SNR = 10 * log10(Ppeak / Pavg)
```
Where Pavg is mean power in ±3 Hz band excluding ±0.5 Hz around peak.

## Usage

### Process Single Experiment
```bash
python wp2_process.py --experiment 007_Fast_stbd_turn_1 --session afternoon
```

### Process with Specific Sensors
```bash
python wp2_process.py --experiment 016_Straight_cruise_1 --session afternoon --sensors Sensor_3 Sensor_wb
```

### Debug Mode
```bash
python wp2_process.py --experiment 026_Engine_rpm_sweep --session afternoon --log-level DEBUG
```

### Custom Output Directory
```bash
python wp2_process.py --experiment 007_Fast_stbd_turn_1 --session afternoon --output /path/to/results
```

## Output Structure

```
results/
└── wp2/
    ├── morning/
    │   └── 015_Skirt_shift_turns_Sensor_3_rpm.h5
    ├── afternoon/
    │   ├── 007_Fast_stbd_turn_1_Sensor_3_rpm.h5
    │   └── 026_Engine_rpm_sweep_Sensor_wb_rpm.h5
    ├── plots/
    │   ├── morning/
    │   └── afternoon/
    │       └── 007_Fast_stbd_turn_1_Sensor_3_diagnostic.png
    └── wp2_done.flag
```

## HDF5 Output Format

Each output file contains:
```
/rpm_estimation/
├── time            # Timestamps (float64)
├── rpm             # RPM estimates (float64)
├── snr_db          # SNR values in dB (float64)
├── valid           # Validity flags (bool)
├── harmonics/      # Harmonic amplitudes
│   ├── 1           # Fundamental
│   ├── 2           # 2nd harmonic
│   └── ...
└── statistics/     # Summary statistics (attributes)
    ├── mean_rpm
    ├── availability_percent
    ├── total_frames
    └── valid_frames
```

## Diagnostic Plots

Each experiment generates a 3-panel diagnostic plot:
1. **RPM over time**: Shows valid (blue) and invalid (red) estimates
2. **SNR over time**: Shows SNR with threshold line
3. **Example PSD**: Shows power spectrum from middle of data

## Success Criteria

✅ Synthetic 25 Hz sine wave → 1500 ± 5 RPM with SNR > 25 dB  
✅ Engine idle detected at 700-800 RPM in static experiments  
✅ SNR > 10 dB for > 80% of frames in normal operation  
✅ Harmonic ratios consistent (2:1, 3:1, etc.)  
✅ Processing time < 30s per experiment  

## Validation Results

### Test Suite
Run unit tests:
```bash
pytest tests/test_spectral.py -v
```

Test coverage includes:
- Welch PSD computation
- Peak detection algorithms
- SNR calculation
- Harmonic extraction
- Full RPM extraction pipeline

### Expected Performance
- **Static experiments**: Stable RPM around 700-800
- **Dynamic maneuvers**: RPM varies 700-2400
- **Engine sweep (026)**: Clear RPM ramp visible

## Known Limitations

1. **Minimum data length**: Requires 6 seconds for one estimate
2. **Frequency resolution**: ±5 RPM due to 0.167 Hz bins
3. **Low SNR handling**: Estimates unreliable below 10 dB SNR
4. **Harmonic confusion**: May misidentify fundamental in extreme cases

## Next Steps

After WP-2 completion:
- **WP-3**: Implement STFT for better temporal resolution
- **WP-4**: Multi-sensor fusion based on SNR
- **WP-5**: Validation against ground truth
- **WP-6**: Batch processing of all experiments

## Troubleshooting

### No peaks detected
- Check if data has sufficient vibration amplitude
- Verify high-pass filtering didn't remove signal
- Try lowering noise_floor_db threshold

### Wrong RPM values
- Check harmonic relationships in diagnostic plots
- Verify frequency search range (10-50 Hz)
- Examine PSD for unexpected peaks

### Low availability
- Check SNR values in diagnostic plots
- May need to adjust SNR threshold
- Consider sensor mounting issues