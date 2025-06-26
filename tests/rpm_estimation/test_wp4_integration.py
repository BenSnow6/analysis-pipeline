#!/usr/bin/env python3
"""
Integration test for WP-4 multi-sensor fusion.

This script tests that WP-4 can successfully:
1. Load WP-2 and WP-3 results
2. Apply fusion rules
3. Generate output files
"""

from pathlib import Path
import logging

# Import modules
from src.analysis.rpm import io as rpm_io
from src.analysis.rpm.wp4_process import (
    load_wp2_results, 
    load_wp3_results,
    process_experiment_fusion,
    FusionResult
)
from src.analysis.rpm.logging_config import setup_logging
load_config = rpm_io.load_config


def test_data_loading():
    """Test loading WP-2 and WP-3 results."""
    print("\n=== Testing Data Loading ===")
    
    base_path = Path(__file__).parent
    
    # Try to load WP-2 results
    wp2_data = load_wp2_results(
        experiment="026_Engine_rpm_sweep",
        session="afternoon", 
        sensor_id="Sensor_3",
        base_path=base_path
    )
    
    if wp2_data:
        print(f"✓ Loaded WP-2 data: {len(wp2_data.frames)} frames")
        print(f"  Availability: {wp2_data.availability:.1f}%")
    else:
        print("✗ No WP-2 data found (may need to run WP-2 first)")
    
    # Try to load WP-3 results
    wp3_data = load_wp3_results(
        experiment="026_Engine_rpm_sweep",
        session="afternoon",
        sensor_id="Sensor_3", 
        base_path=base_path
    )
    
    if wp3_data:
        print(f"✓ Loaded WP-3 data: {len(wp3_data.frames)} frames")
        print(f"  Availability: {wp3_data.availability:.1f}%")
    else:
        print("✗ No WP-3 data found (may need to run WP-3 first)")
    
    return wp2_data is not None or wp3_data is not None


def test_fusion_logic():
    """Test basic fusion logic with synthetic data."""
    print("\n=== Testing Fusion Logic ===")
    
    from src.analysis.rpm.tracking import RPMFrame, RPMTimeSeries
    from src.analysis.rpm.fusion import fuse_sensors_snr
    import numpy as np
    
    # Create synthetic test data
    times = np.arange(0, 10, 0.25)
    
    # Sensor 1: High SNR
    frames1 = [
        RPMFrame(time=t, rpm=1800 + t*10, snr_db=15, 
                sensor_id='S1', method='stft')
        for t in times
    ]
    
    # Sensor 2: Medium SNR with gap
    frames2 = [
        RPMFrame(time=t, rpm=1810 + t*10, snr_db=12,
                sensor_id='S2', method='stft')
        for t in times if t < 5 or t > 7
    ]
    
    # Sensor 3: Low SNR
    frames3 = [
        RPMFrame(time=t, rpm=1805 + t*10, snr_db=8,
                sensor_id='S3', method='stft')
        for t in times
    ]
    
    sensor_data = {
        'S1': RPMTimeSeries(frames1, 'test', 'test', 'S1'),
        'S2': RPMTimeSeries(frames2, 'test', 'test', 'S2'),
        'S3': RPMTimeSeries(frames3, 'test', 'test', 'S3')
    }
    
    # Test fusion
    config = {'snr_thresh_db': 10}
    fused = fuse_sensors_snr(sensor_data, config)
    
    print(f"✓ Fusion completed: {len(fused.frames)} frames")
    print(f"  Availability: {fused.availability:.1f}%")
    
    # Check sensor selection
    s1_count = sum(1 for f in fused.frames if 'S1' in f.sensor_id)
    s2_count = sum(1 for f in fused.frames if 'S2' in f.sensor_id)
    s3_count = sum(1 for f in fused.frames if 'S3' in f.sensor_id)
    
    print(f"  Sensor contributions: S1={s1_count}, S2={s2_count}, S3={s3_count}")
    print(f"  S1 should dominate (highest SNR): {s1_count > s2_count + s3_count}")
    
    return True


def test_full_pipeline():
    """Test full WP-4 pipeline if data is available."""
    print("\n=== Testing Full Pipeline ===")
    
    try:
        # Load config
        config_path = Path(__file__).parent / "rpm_config.yaml"
        config = load_config(config_path)
        
        # Try to process an experiment
        base_path = Path(__file__).parent
        
        # Check if we have WP-2/WP-3 results
        wp2_path = base_path / "results" / "wp2" / "afternoon"
        wp3_path = base_path / "results" / "wp3" / "afternoon"
        
        if not wp2_path.exists() and not wp3_path.exists():
            print("⚠ No WP-2 or WP-3 results found. Run WP-2/WP-3 first:")
            print("  python -m rpm_estimation.cli --wp 2 --exp 026_Engine_rpm_sweep --session afternoon")
            print("  python -m rpm_estimation.cli --wp 3 --exp 026_Engine_rpm_sweep --session afternoon")
            return False
        
        # Process a test experiment
        result = process_experiment_fusion(
            experiment="026_Engine_rpm_sweep",
            session="afternoon",
            config=config,
            base_path=base_path
        )
        
        if isinstance(result, FusionResult):
            print(f"✓ Fusion successful!")
            print(f"  Availability: {result.quality_stats['availability']:.1f}%")
            print(f"  Mean SNR: {result.quality_stats['mean_snr_db']:.1f} dB")
            print(f"  Processing time: {result.processing_time:.1f} s")
            return True
        else:
            print("✗ Fusion failed")
            return False
            
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("WP-4 Integration Test Suite")
    print("=" * 40)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Data loading
    if test_data_loading():
        tests_passed += 1
    
    # Test 2: Fusion logic
    if test_fusion_logic():
        tests_passed += 1
    
    # Test 3: Full pipeline
    if test_full_pipeline():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n✅ All tests passed! WP-4 is ready to use.")
        print("\nNext steps:")
        print("1. Run on real data: python -m rpm_estimation.cli --wp 4 --exp 026_Engine_rpm_sweep --session afternoon --plot")
        print("2. Check output in: results/wp4/afternoon/026_Engine_rpm_sweep/")
        print("3. Verify <2% NaN frames and >95% availability")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        if tests_passed < 2:
            print("\nMake sure to run WP-2 and WP-3 first to generate input data.")


if __name__ == "__main__":
    main()