# Failing Tests Analysis

This document tracks tests that are currently failing in the CI/CD pipeline after the repository reorganization (2025-06-26).

## Summary

After the repository reorganization, most tests pass successfully:
- **Python 3.8**: 102 passed, 13 failed, 2 skipped
- **Python 3.9**: 102 passed, 13 failed, 2 skipped  
- **Python 3.10**: 168 passed, 17 failed, 2 skipped

The failures are not related to the reorganization but are functional issues with the tests themselves.

## Failing Tests by Category

### 1. Configuration File Issues

#### `tests/rpm_estimation/test_cli.py::TestArgumentValidation::test_output_directory_creation`
- **Error**: `FileNotFoundError: Configuration file not found: rpm_config.yaml`
- **Reason**: Test expects a default config file in the current directory
- **Fix needed**: Update test to create a temporary config file or mock the config loading

### 2. Test Logic/Assertion Issues

#### `tests/rpm_estimation/test_fusion.py::TestSensorSelection::test_select_best_sensor_invalid_excluded`
- **Error**: `AssertionError: assert 'S3' == 'S2'`
- **Reason**: Test expects sensor S2 to be selected, but logic selects S3
- **Fix needed**: Review fusion selection logic or update test expectations

#### `tests/rpm_estimation/test_fusion.py::TestSensorSelection::test_select_best_sensor_no_valid`
- **Error**: `AssertionError: assert RPMFrame(...) is None`
- **Reason**: Test expects None when no valid sensors, but returns a frame with NaN RPM
- **Fix needed**: Update fusion logic to return None or update test expectations

#### `tests/rpm_estimation/test_fusion.py::TestInterpolation::test_interpolate_small_gaps`
- **Error**: `AssertionError: assert 100.0 > 100.0`
- **Reason**: Test expects availability to increase after interpolation, but it remains 100%
- **Fix needed**: Review interpolation logic for already-complete data

### 3. Quality Assessment Issues

#### `tests/rpm_estimation/test_quality.py::TestWindowProcessing::test_pad_strategy`
- **Error**: `assert False == True`
- **Reason**: Padding strategy not implemented as expected
- **Fix needed**: Implement proper padding strategy or update test

#### `tests/rpm_estimation/test_quality.py::TestSignalQualityAssessment::test_quality_assessment`
- **Error**: `assert 0 >= 1`
- **Reason**: Quality metrics not being calculated correctly
- **Fix needed**: Fix quality metric calculation

#### `tests/rpm_estimation/test_quality.py::test_quality_report_generation`
- **Error**: `assert 0 > 0` (empty report)
- **Reason**: Report generation returns empty list
- **Fix needed**: Implement report generation properly

### 4. Schema Validation Issues

#### `tests/rpm_estimation/test_schema.py::TestSchemaValidation::test_valid_schema`
- **Error**: `assert False == True`
- **Reason**: Schema validation not implemented correctly
- **Fix needed**: Fix schema validation logic

#### `tests/rpm_estimation/test_schema.py::TestSchemaValidation::test_null_detection`
- **Error**: `assert False` (no nulls detected when expected)
- **Reason**: Null detection logic not working
- **Fix needed**: Implement proper null detection

### 5. Spectral Analysis Issues

#### `tests/rpm_estimation/test_spectral.py::TestRPMExtraction::test_no_peaks`
- **Error**: `AssertionError: assert RPMFrame(...) is None`
- **Reason**: Test expects None for noise, but peak detection finds spurious peaks
- **Fix needed**: Improve peak detection threshold or update test

### 6. STFT Processing Issues

#### `tests/rpm_estimation/test_stft.py::TestSTFTCore::test_edge_handling`
- **Error**: `assert not np.array_equal(...)`
- **Reason**: Edge handling produces identical output to input
- **Fix needed**: Implement proper edge handling in STFT

#### `tests/rpm_estimation/test_stft.py::TestRPMExtraction::test_triangular_ramp`
- **Error**: `assert 1160.0 < 750` (varies between runs)
- **Reason**: RPM extraction from ramp signal not working as expected
- **Fix needed**: Fix ramp signal RPM extraction or adjust test tolerance

#### `tests/rpm_estimation/test_stft.py::TestAntiAliasVerification::test_verify_filter_high_peaks`
- **Error**: `assert any('peak-to-RMS' in w for w in warnings)`
- **Reason**: Expected warning not generated
- **Fix needed**: Implement anti-alias warning generation

### 7. Orientation Module Issues

#### `tests/orientation/test_orientation.py::TestStaticDetector::test_perfect_static_detection`
- **Error**: `AssertionError: 9.995 != 10.0`
- **Reason**: Floating point precision issue in time calculation
- **Fix needed**: Use approximate comparison with tolerance

#### `tests/orientation/test_orientation.py::TestOrientationIntegration::test_validate_sensor_synthetic`
- **Error**: `AssertionError: 90.0 not less than 1.0`
- **Reason**: Orientation validation returning 90° error instead of expected small error
- **Fix needed**: Fix orientation calculation or update test data

### 8. Data Availability Issues

#### `tests/config/test_experiment_manifest_integration.py::TestFilesystemIntegrity::test_data_subdirectories_exist`
- **Error**: Missing 60 data subdirectories
- **Reason**: Test data not included in CI environment
- **Fix needed**: Skip test in CI or provide minimal test data

## Recommendations

1. **Priority 1**: Fix configuration and basic logic issues (fusion, quality, schema)
2. **Priority 2**: Address numerical/algorithmic issues (STFT, spectral analysis)
3. **Priority 3**: Handle data availability and CI-specific issues
4. **Priority 4**: Fine-tune numerical tolerances and edge cases

Most of these failures appear to be long-standing issues with the test suite rather than problems introduced by the reorganization. The repository structure changes have been successful, with all imports and paths working correctly.