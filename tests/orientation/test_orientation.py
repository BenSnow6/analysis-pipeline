"""
Unit tests for orientation analysis modules.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.analysis.orientation.static_detector import StaticDetector
from src.analysis.orientation.rotation_validator import RotationValidator
from src.analysis.orientation.dynamic_validator import DynamicValidator
from src.analysis.orientation.bias_estimator import BiasEstimator
from src.analysis.orientation.orientation_check import OrientationChecker


class TestStaticDetector(unittest.TestCase):
    """Test static segment detection."""
    
    def setUp(self):
        self.detector = StaticDetector()
        
    def test_perfect_static_detection(self):
        """Test detection of perfectly static data."""
        # Create 10 seconds of data at 200 Hz
        duration = 10.0
        rate = 200.0
        n_samples = int(duration * rate)
        
        timestamp = np.arange(n_samples) / rate
        
        # Perfect static: zero gyro, constant accel (gravity)
        gyro = np.zeros((n_samples, 3))
        accel = np.ones((n_samples, 3)) * 9.81
        accel[:, :2] = 0  # Only Z component
        
        segments = self.detector.detect_static_segments(timestamp, gyro, accel, rate)
        
        # Should detect one long static segment
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0][0], 0.0, places=2)
        # Allow for edge effects from windowing (up to 1 sample at 200Hz = 0.005s)
        self.assertAlmostEqual(segments[0][1], duration, places=1)
        
    def test_dynamic_data_rejection(self):
        """Test that dynamic data is not detected as static."""
        duration = 5.0
        rate = 200.0
        n_samples = int(duration * rate)
        
        timestamp = np.arange(n_samples) / rate
        
        # High rotation rate
        gyro = np.ones((n_samples, 3)) * 0.5  # rad/s
        accel = np.ones((n_samples, 3)) * 9.81
        
        segments = self.detector.detect_static_segments(timestamp, gyro, accel, rate)
        
        # Should not detect any static segments
        self.assertEqual(len(segments), 0)
        
    def test_mixed_static_dynamic(self):
        """Test detection with mixed static and dynamic periods."""
        duration = 10.0
        rate = 200.0
        n_samples = int(duration * rate)
        
        timestamp = np.arange(n_samples) / rate
        gyro = np.zeros((n_samples, 3))
        accel = np.ones((n_samples, 3)) * 9.81
        
        # Add dynamic period in the middle (3-7 seconds)
        dynamic_start = int(3 * rate)
        dynamic_end = int(7 * rate)
        gyro[dynamic_start:dynamic_end, :] = 0.1  # Above threshold
        
        segments = self.detector.detect_static_segments(timestamp, gyro, accel, rate)
        
        # Should detect two static segments
        self.assertEqual(len(segments), 2)
        
        # First segment: 0-3 seconds (minus window effects)
        self.assertLess(segments[0][0], 1.0)
        self.assertGreater(segments[0][1], 2.0)
        
        # Second segment: 7-10 seconds (minus window effects)
        self.assertLess(segments[1][0], 8.0)
        self.assertGreater(segments[1][1], 9.0)


class TestRotationValidator(unittest.TestCase):
    """Test rotation matrix validation."""
    
    def setUp(self):
        self.validator = RotationValidator()
        
    def test_identity_matrix_validation(self):
        """Test validation of identity matrix."""
        R = np.eye(3)
        result = self.validator.validate_rotation_matrix(R)
        
        self.assertTrue(result['valid'])
        self.assertLess(result['orthonormality_error'], 1e-10)
        self.assertAlmostEqual(result['determinant'], 1.0)
        self.assertTrue(result['is_proper_rotation'])
        
    def test_invalid_matrix(self):
        """Test detection of invalid rotation matrix."""
        # Non-orthogonal matrix
        R = np.array([[1, 0.5, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        
        result = self.validator.validate_rotation_matrix(R)
        self.assertFalse(result['valid'])
        
    def test_gravity_extraction(self):
        """Test gravity direction extraction."""
        # Create static accelerometer data pointing down
        n_samples = 100
        gravity_mag = 9.81
        
        # Gravity in sensor Z direction
        accel_static = np.zeros((n_samples, 3))
        accel_static[:, 2] = gravity_mag
        
        # Add small noise
        accel_static += np.random.normal(0, 0.01, accel_static.shape)
        
        gravity_dir = self.validator.extract_gravity_direction(accel_static)
        
        # Should be close to [0, 0, 1]
        expected = np.array([0, 0, 1])
        angle_error = np.arccos(np.clip(np.dot(gravity_dir, expected), -1, 1))
        
        self.assertLess(angle_error, 0.01)  # Less than 0.01 radians error


class TestDynamicValidator(unittest.TestCase):
    """Test dynamic maneuver validation."""
    
    def setUp(self):
        self.validator = DynamicValidator()
        
    def test_acceleration_phase_detection(self):
        """Test detection of acceleration phases."""
        duration = 10.0
        rate = 200.0
        n_samples = int(duration * rate)
        
        timestamp = np.arange(n_samples) / rate
        accel_body = np.zeros((n_samples, 3))
        
        # Add gravity
        accel_body[:, 2] = 9.81
        
        # Add forward acceleration from 2-6 seconds
        accel_start = int(2 * rate)
        accel_end = int(6 * rate)
        accel_body[accel_start:accel_end, 0] = 1.0  # 1 m/s² forward
        
        phases = self.validator.find_acceleration_phases(accel_body, timestamp)
        
        self.assertEqual(len(phases), 1)
        self.assertAlmostEqual(phases[0][0], 2.0, places=1)
        self.assertAlmostEqual(phases[0][1], 6.0, places=1)
        
    def test_turn_phase_detection(self):
        """Test detection of turn phases."""
        duration = 10.0
        rate = 200.0
        n_samples = int(duration * rate)
        
        timestamp = np.arange(n_samples) / rate
        gyro_body = np.zeros((n_samples, 3))
        
        # Add yaw rate from 3-7 seconds
        turn_start = int(3 * rate)
        turn_end = int(7 * rate)
        gyro_body[turn_start:turn_end, 2] = 0.2  # rad/s (positive = starboard)
        
        turns = self.validator.find_turn_phases(gyro_body, timestamp)
        
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]['direction'], 'starboard')
        self.assertAlmostEqual(turns[0]['mean_rate'], 0.2, places=2)
        self.assertAlmostEqual(turns[0]['duration'], 4.0, places=1)


class TestBiasEstimator(unittest.TestCase):
    """Test sensor bias estimation."""
    
    def setUp(self):
        self.estimator = BiasEstimator()
        
    def test_zero_bias_estimation(self):
        """Test estimation with zero bias."""
        duration = 30.0
        rate = 200.0
        n_samples = int(duration * rate)
        
        timestamp = np.arange(n_samples) / rate
        
        # Perfect IMU: gravity only, zero gyro
        accel = np.zeros((n_samples, 3))
        accel[:, 2] = 9.80665  # Gravity in Z
        gyro = np.zeros((n_samples, 3))
        
        # Identity rotation matrix
        R_bs = np.eye(3)
        
        results = self.estimator.estimate_biases(
            'test_sensor', accel, gyro, timestamp, R_bs
        )
        
        # Biases should be near zero
        self.assertLess(np.linalg.norm(results['accel_bias_sensor_m_s2']), 0.01)
        self.assertLess(np.linalg.norm(results['gyro_bias_sensor_rad_s']), 0.001)
        
    def test_bias_estimation_with_offset(self):
        """Test estimation with known bias."""
        duration = 30.0
        rate = 200.0
        n_samples = int(duration * rate)
        
        timestamp = np.arange(n_samples) / rate
        
        # Add known biases
        accel_bias_true = np.array([0.1, -0.05, 0.2])
        gyro_bias_true = np.array([0.01, -0.005, 0.002])
        
        # IMU with biases
        accel = np.zeros((n_samples, 3))
        accel[:, 2] = 9.80665  # Gravity
        accel += accel_bias_true  # Add bias
        
        gyro = np.zeros((n_samples, 3)) + gyro_bias_true
        
        # Identity rotation matrix
        R_bs = np.eye(3)
        
        results = self.estimator.estimate_biases(
            'test_sensor', accel, gyro, timestamp, R_bs
        )
        
        # Check estimated biases are close to true values
        accel_bias_est = results['accel_bias_sensor_m_s2']
        gyro_bias_est = results['gyro_bias_sensor_rad_s']
        
        np.testing.assert_allclose(accel_bias_est, accel_bias_true, atol=0.01)
        np.testing.assert_allclose(gyro_bias_est, gyro_bias_true, atol=0.001)


class TestOrientationIntegration(unittest.TestCase):
    """Integration tests for complete orientation validation."""
    
    def setUp(self):
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        config = {
            'craft': {
                'dimensions': {'length_m': 13.25, 'beam_m': 6.18, 'height_m': 4.90},
                'frame': {'origin': 'CG', 'x_axis': 'Forward', 'y_axis': 'Starboard', 'z_axis': 'Down'}
            },
            'sensors': {
                'Sensor_3': {
                    'position_m': [0, 2.5, 1.2],
                    'expected_axes': {'x_direction': 'Upward', 'y_direction': 'Forward', 'z_direction': 'Port'},
                    'type': 'primary'
                }
            },
            'physics': {
                'gravity_m_s2': 9.80665,
                'gravity_body_frame': [0, 0, 9.80665]
            },
            'validation': {
                'tolerances': {
                    'primary_sensors_deg': 3.0,
                    'secondary_sensors_deg': 5.0,
                    'orthogonality_threshold': 0.001,
                    'cross_sensor_dcm_deg': 2.0
                }
            },
            'static_detection': {
                'gyro_threshold_rad_s': 0.05,
                'accel_std_threshold_m_s2': 0.05,
                'min_duration_s': 1.0,
                'window_size_s': 1.0
            },
            'bias_estimation': {
                'static_duration_s': 30.0,
                'outlier_threshold_sigma': 3.0
            },
            'maneuver_validation': {
                'experiments': {
                    'test_experiment': {
                        'description': 'Test maneuver',
                        'expected_pattern': 'Test pattern'
                    }
                }
            },
            'plotting': {
                'figure_size': [10, 8],
                'gravity_vector_color': 'red',
                'sensor_axes_colors': {'x': 'blue', 'y': 'green', 'z': 'orange'},
                'vector_scale': 0.3,
                'save_format': 'png',
                'dpi': 150
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
            
        self.checker = OrientationChecker(self.config_path)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_validate_sensor_synthetic(self):
        """Test complete sensor validation with synthetic data."""
        # Create synthetic data
        duration = 40.0
        rate = 200.0
        n_samples = int(duration * rate)
        timestamp = np.arange(n_samples) / rate
        
        # Sensor 3 orientation: X up, Y forward, Z port
        # This means gravity should appear as negative X in sensor frame
        accel = np.zeros((n_samples, 3))
        accel[:, 0] = -9.80665  # Gravity appears in negative X
        
        gyro = np.zeros((n_samples, 3))
        
        sensor_data = {
            'accel': accel,
            'gyro': gyro
        }
        
        results = self.checker.validate_sensor(
            'Sensor_3', sensor_data, timestamp, 'test_experiment'
        )
        
        # Check results
        self.assertIn('rotation_validation', results)
        self.assertIn('bias_estimation', results)
        self.assertLess(results['rotation_error_deg'], 1.0)  # Should be very small
        self.assertTrue(results['static_valid'])


if __name__ == '__main__':
    unittest.main()