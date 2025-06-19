"""
Tests for configuration loading and management.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from rpm_estimation.io import load_config, save_config


class TestConfigOperations:
    """Test configuration file operations."""
    
    def test_load_default_config(self):
        """Test loading the default configuration file."""
        # Load config from module directory
        module_dir = Path(__file__).parent.parent
        config_path = module_dir / 'rpm_config.yaml'
        
        if config_path.exists():
            config = load_config(config_path)
            
            # Check required top-level keys
            assert 'fs' in config
            assert config['fs'] == 200
            
            assert 'hp_cutoff' in config
            assert config['hp_cutoff'] == 5
            
            assert 'welch' in config
            assert 'stft' in config
            assert 'snr_thresh_db' in config
    
    def test_config_structure(self):
        """Test the structure of configuration."""
        # Create test config
        test_config = {
            'fs': 200,
            'hp_cutoff': 5,
            'welch': {
                'win_sec': 6,
                'overlap': 0.5,
                'window': 'hann',
                'detrend': 'linear'
            },
            'stft': {
                'win_sec': 1.0,
                'hop_sec': 0.25,
                'window': 'hann'
            },
            'snr_thresh_db': 10,
            'anti_alias': {
                'cutoff_hz': 85,
                'order': 4,
                'type': 'butterworth'
            }
        }
        
        # Validate structure
        assert test_config['fs'] > 0
        assert 0 < test_config['welch']['overlap'] < 1
        assert test_config['welch']['win_sec'] > test_config['stft']['win_sec']
        assert test_config['snr_thresh_db'] > 0
    
    def test_round_trip_save_load(self):
        """Test saving and loading configuration."""
        # Create test config
        test_config = {
            'fs': 250,
            'hp_cutoff': 7,
            'welch': {
                'win_sec': 8,
                'overlap': 0.75
            },
            'test_field': 'test_value'
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save config
            save_config(test_config, temp_path)
            
            # Load it back
            loaded_config = load_config(temp_path)
            
            # Verify contents match
            assert loaded_config['fs'] == test_config['fs']
            assert loaded_config['hp_cutoff'] == test_config['hp_cutoff']
            assert loaded_config['welch']['win_sec'] == test_config['welch']['win_sec']
            assert loaded_config['welch']['overlap'] == test_config['welch']['overlap']
            assert loaded_config['test_field'] == test_config['test_field']
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_validate_welch_params(self):
        """Test validation of Welch parameters."""
        config = {
            'fs': 200,
            'welch': {
                'win_sec': 6,
                'overlap': 0.5
            }
        }
        
        # Calculate derived parameters
        nperseg = int(config['welch']['win_sec'] * config['fs'])
        noverlap = int(config['welch']['overlap'] * nperseg)
        
        assert nperseg == 1200  # 6 seconds at 200 Hz
        assert noverlap == 600   # 50% overlap
        
        # Frequency resolution
        freq_resolution = config['fs'] / nperseg
        assert freq_resolution == pytest.approx(0.167, rel=0.01)  # ~0.167 Hz
        
        # RPM resolution
        rpm_resolution = freq_resolution * 60
        assert rpm_resolution == pytest.approx(10.0, rel=0.01)  # ~10 RPM
    
    def test_validate_stft_params(self):
        """Test validation of STFT parameters."""
        config = {
            'fs': 200,
            'stft': {
                'win_sec': 1.0,
                'hop_sec': 0.25
            }
        }
        
        # Calculate derived parameters
        nperseg = int(config['stft']['win_sec'] * config['fs'])
        hop_length = int(config['stft']['hop_sec'] * config['fs'])
        overlap = nperseg - hop_length
        
        assert nperseg == 200     # 1 second at 200 Hz
        assert hop_length == 50   # 0.25 second hop
        assert overlap == 150     # 75% overlap
        
        # Time resolution
        time_resolution = config['stft']['hop_sec']
        assert time_resolution == 0.25  # 4 frames per second
    
    def test_frequency_range_params(self):
        """Test frequency range parameters for RPM."""
        config = {
            'frequency': {
                'min_rpm': 600,
                'max_rpm': 3000,
                'min_freq_hz': 10,
                'max_freq_hz': 50
            }
        }
        
        # Check conversions
        min_freq_calc = config['frequency']['min_rpm'] / 60
        max_freq_calc = config['frequency']['max_rpm'] / 60
        
        assert min_freq_calc == config['frequency']['min_freq_hz']
        assert max_freq_calc == config['frequency']['max_freq_hz']
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path('nonexistent_config.yaml'))