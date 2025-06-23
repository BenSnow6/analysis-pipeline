"""
Tests for CLI functionality.
"""

import pytest
from pathlib import Path
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import create_parser, validate_args


class TestCLIParser:
    """Test CLI argument parsing."""
    
    def test_basic_arguments(self):
        """Test parsing of basic arguments."""
        parser = create_parser()
        
        # Test valid arguments
        args = parser.parse_args([
            '--wp', '1',
            '--exp', '007_Fast_stbd_turn_1',
            '--session', 'afternoon'
        ])
        
        assert args.wp == 1
        assert args.exp == '007_Fast_stbd_turn_1'
        assert args.session == 'afternoon'
    
    def test_work_package_choices(self):
        """Test work package validation."""
        parser = create_parser()
        
        # Valid WP
        args = parser.parse_args(['--wp', '1'])
        assert args.wp == 1
        
        # Invalid WP should raise
        with pytest.raises(SystemExit):
            parser.parse_args(['--wp', '7'])
    
    def test_sensor_override(self):
        """Test sensor selection override."""
        parser = create_parser()
        
        args = parser.parse_args([
            '--wp', '1',
            '--sensors', 'Sensor_3', 'Sensor_5'
        ])
        
        assert args.sensors == ['Sensor_3', 'Sensor_5']
    
    def test_batch_processing_flags(self):
        """Test batch processing arguments."""
        parser = create_parser()
        
        # Test --all flag
        args = parser.parse_args(['--wp', '1', '--all', '--session', 'morning'])
        assert args.all == True
        assert args.session == 'morning'
        
        # Test --list flag
        args = parser.parse_args(['--wp', '1', '--list'])
        assert args.list == True
    
    def test_logging_options(self):
        """Test logging configuration options."""
        parser = create_parser()
        
        args = parser.parse_args([
            '--wp', '1',
            '--log-level', 'DEBUG',
            '--log-format', 'json',
            '--log-file', 'test.log'
        ])
        
        assert args.log_level == 'DEBUG'
        assert args.log_format == 'json'
        assert args.log_file == Path('test.log')
    
    def test_validation_flags(self):
        """Test validation-related flags."""
        parser = create_parser()
        
        args = parser.parse_args([
            '--wp', '1',
            '--validate',
            '--include-synthetic'
        ])
        
        assert args.validate == True
        assert args.include_synthetic == True
    
    def test_processing_options(self):
        """Test processing option flags."""
        parser = create_parser()
        
        args = parser.parse_args([
            '--wp', '1',
            '--no-parallel',
            '--dry-run'
        ])
        
        assert args.no_parallel == True
        assert args.dry_run == True


class TestArgumentValidation:
    """Test argument validation logic."""
    
    def test_config_file_validation(self, tmp_path):
        """Test configuration file path validation."""
        parser = create_parser()
        
        # Create temporary config
        config_path = tmp_path / 'test_config.yaml'
        config_path.write_text("fs: 200\n")
        
        args = parser.parse_args([
            '--wp', '1',
            '--config', str(config_path)
        ])
        
        # Should not raise
        validated = validate_args(args)
        assert validated.config == config_path
    
    def test_default_config_fallback(self):
        """Test fallback to module directory config."""
        parser = create_parser()
        
        # Use non-existent config
        args = parser.parse_args([
            '--wp', '1',
            '--config', 'nonexistent.yaml'
        ])
        
        # If module config exists, should use it
        module_dir = Path(__file__).parent.parent
        expected_config = module_dir / 'rpm_config.yaml'
        
        if expected_config.exists():
            validated = validate_args(args)
            assert validated.config == expected_config
        else:
            with pytest.raises(FileNotFoundError):
                validate_args(args)
    
    def test_output_directory_creation(self, tmp_path):
        """Test output directory is created if needed."""
        parser = create_parser()
        
        output_dir = tmp_path / 'new_output_dir'
        args = parser.parse_args([
            '--wp', '1',
            '--output-dir', str(output_dir)
        ])
        
        # Directory shouldn't exist yet
        assert not output_dir.exists()
        
        # Validation should create it
        validate_args(args)
        assert output_dir.exists()


def test_wp1_argument_combinations():
    """Test valid argument combinations for WP-1."""
    parser = create_parser()
    
    # Valid: single experiment
    args = parser.parse_args([
        '--wp', '1',
        '--exp', 'test_exp',
        '--session', 'morning'
    ])
    assert args.exp == 'test_exp'
    assert args.session == 'morning'
    
    # Valid: all experiments for session
    args = parser.parse_args([
        '--wp', '1',
        '--all',
        '--session', 'afternoon'
    ])
    assert args.all == True
    assert args.session == 'afternoon'
    
    # Valid: list experiments
    args = parser.parse_args([
        '--wp', '1',
        '--list'
    ])
    assert args.list == True
    
    # Valid: validation mode
    args = parser.parse_args([
        '--wp', '1',
        '--validate'
    ])
    assert args.validate == True


def test_help_message():
    """Test help message generation."""
    parser = create_parser()
    
    # Should not raise
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(['--help'])
    
    # Help exits with 0
    assert exc_info.value.code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])