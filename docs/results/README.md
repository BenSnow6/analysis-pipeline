# Results Directory Structure

This directory contains all analysis results organized by thesis work packages.

## Directory Structure

- **raw_data_analysis/** - Analysis of raw experimental data quality and characteristics
- **timestamp_analysis/** - Results from timestamp consistency analysis
- **alignment/** - Time alignment results and validation
- **orientation/** - Sensor orientation analysis and validation results
- **attitude_estimation/** - Pitch and roll estimation results
- **steering/** - Steering angle estimation results
- **rpm_estimation/** - Engine RPM estimation work
  - wp0_exploration/ - Initial exploration and feasibility studies
  - wp1_preprocessing/ - Data preprocessing results
  - wp2_peak_detection/ - Peak detection algorithm results
  - wp3_stft/ - Short-Time Fourier Transform results
  - wp4_fusion/ - Multi-sensor fusion results
  - wp5-7_future/ - Future work (wavelet, ML, adaptive methods)
- **validation/** - Overall system validation results

## Notes

- Actual processed data files remain in `/data/processed/`
- This directory contains reports, plots, and summary documents
- Each subdirectory should have its own README explaining the specific contents