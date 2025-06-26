"""
Shared pytest fixtures and configuration for all tests.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import yaml
from typing import Dict


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(project_root) -> Path:
    """Get the config directory."""
    return project_root / "config"


@pytest.fixture
def manifest_path(config_dir) -> Path:
    """Get the path to the experiment manifest."""
    return config_dir / "experiments" / "experiment_manifest.yaml"


@pytest.fixture
def test_data_dir(tmp_path) -> Path:
    """Create a temporary test data directory structure."""
    # Create base structure
    data_root = tmp_path / "data" / "raw"
    data_root.mkdir(parents=True)
    
    # Create morning and afternoon directories
    morning_dir = data_root / "morning" / "Experiments"
    afternoon_dir = data_root / "afternoon" / "Experiments"
    morning_dir.mkdir(parents=True)
    afternoon_dir.mkdir(parents=True)
    
    return data_root


@pytest.fixture
def mock_experiment_structure(test_data_dir) -> Dict[str, Path]:
    """Create a mock experiment directory structure for testing."""
    experiments = {}
    
    # Create morning experiment
    morning_exp = test_data_dir / "morning" / "Experiments" / "006_Departure"
    morning_exp.mkdir(parents=True)
    
    # Create GPS data
    gps_dir = morning_exp / "GPS"
    gps_dir.mkdir()
    (gps_dir / "GPS_006_Departure.csv").write_text("timestamp,lat,lon\n1,0,0\n")
    
    # Create IMU data
    imu_dir = morning_exp / "IMU"
    imu_dir.mkdir()
    
    for sensor in ["Sensor_3", "Sensor_4", "Sensor_5"]:
        sensor_dir = imu_dir / sensor
        sensor_dir.mkdir()
        (sensor_dir / f"accel_006_Departure.csv").write_text("timestamp,x,y,z\n1,0,0,0\n")
        (sensor_dir / f"gyro_006_Departure.csv").write_text("timestamp,x,y,z\n1,0,0,0\n")
    
    experiments['morning_departure'] = morning_exp
    
    # Create afternoon experiment
    afternoon_exp = test_data_dir / "afternoon" / "Experiments" / "007_Fast_stbd_turn_1"
    afternoon_exp.mkdir(parents=True)
    
    # Create GPS data
    gps_dir = afternoon_exp / "GPS"
    gps_dir.mkdir()
    (gps_dir / "GPS_007_Fast_stbd_turn_1.csv").write_text("timestamp,lat,lon\n1,0,0\n")
    
    experiments['afternoon_turn'] = afternoon_exp
    
    # Create an orphan directory (not in manifest)
    orphan = test_data_dir / "morning" / "Experiments" / "999_Orphan_Test"
    orphan.mkdir(parents=True)
    experiments['orphan'] = orphan
    
    return experiments


@pytest.fixture
def sample_manifest_content() -> dict:
    """Create sample manifest content for testing."""
    return {
        'evaluation_experiments': {
            'morning': [
                {
                    'name': '006_Departure',
                    'path': '1b_4_Normal_Take_off/morning/006_Departure',
                    'type': 'dynamic',
                    'description': 'Departure from port',
                    'category': '1b_4_Normal_Take_off',
                    'is_static': False,
                    'paths': {
                        'relative': 'morning/Experiments/006_Departure',
                        'full_path': '/data/raw/morning/Experiments/006_Departure'
                    },
                    'data_types': ['GPS', 'Sensor_3', 'Sensor_4', 'Sensor_5']
                }
            ],
            'afternoon': [
                {
                    'name': '007_Fast_stbd_turn_1',
                    'path': '1a_1_Minimum_Radius_Turn/afternoon/007_Fast_stbd_turn_1',
                    'type': 'dynamic',
                    'description': 'Fast starboard turn',
                    'category': '1a_1_Minimum_Radius_Turn',
                    'is_static': False,
                    'paths': {
                        'relative': 'afternoon/Experiments/007_Fast_stbd_turn_1',
                        'full_path': '/data/raw/afternoon/Experiments/007_Fast_stbd_turn_1'
                    },
                    'data_types': ['GPS']
                }
            ]
        },
        'static_experiments': {
            'morning': [],
            'afternoon': []
        },
        'all_experiments': {
            'morning': [
                {'name': '006_Departure'}
            ],
            'afternoon': [
                {'name': '007_Fast_stbd_turn_1'}
            ]
        },
        'analysis_config': {
            'orientation_validation_experiments': {
                'morning': [],
                'afternoon': []
            }
        }
    }


@pytest.fixture
def test_manifest_file(tmp_path, sample_manifest_content) -> Path:
    """Create a test manifest file."""
    manifest_file = tmp_path / "test_manifest.yaml"
    
    # Update paths in the content to use the temp directory
    for section in ['evaluation_experiments', 'static_experiments']:
        for session in ['morning', 'afternoon']:
            if session in sample_manifest_content[section]:
                for exp in sample_manifest_content[section][session]:
                    if 'paths' in exp:
                        # Update full_path to use tmp_path
                        relative = exp['paths']['relative']
                        exp['paths']['full_path'] = str(tmp_path / "data" / "raw" / relative)
    
    with open(manifest_file, 'w') as f:
        yaml.dump(sample_manifest_content, f)
    
    return manifest_file


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (no filesystem access)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (requires filesystem)"
    )
    config.addinivalue_line(
        "markers", "full_validation: mark test for comprehensive validation"
    )


# Skip integration tests by default in CI
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if config.getoption("--no-integration"):
        skip_integration = pytest.mark.skip(reason="Integration tests disabled")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--no-integration",
        action="store_true",
        default=False,
        help="Skip integration tests"
    )
    parser.addoption(
        "--manifest",
        action="store",
        default=None,
        help="Path to experiment manifest to validate"
    )