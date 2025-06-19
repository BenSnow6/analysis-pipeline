"""
Smoke tests to ensure all modules can be imported.
"""

import pytest


def test_all_imports():
    """Ensure all modules can be imported without errors."""
    # Main module
    import rpm_estimation
    
    # Sub-modules
    import rpm_estimation.io
    import rpm_estimation.preprocess
    import rpm_estimation.spectral
    import rpm_estimation.tracking
    import rpm_estimation.fusion
    import rpm_estimation.cli
    
    # Check version
    assert hasattr(rpm_estimation, '__version__')
    assert isinstance(rpm_estimation.__version__, str)


def test_main_exports():
    """Test that main module exports key components."""
    import rpm_estimation
    
    # Check key exports
    assert hasattr(rpm_estimation, 'RPMFrame')
    assert hasattr(rpm_estimation, 'cli_main')
    
    # Test RPMFrame is accessible
    from rpm_estimation import RPMFrame
    assert RPMFrame is not None


def test_cli_imports():
    """Test CLI module imports and parser creation."""
    from rpm_estimation.cli import create_parser, main
    
    # Test parser creation
    parser = create_parser()
    assert parser is not None
    
    # Test help doesn't crash
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(['--help'])
    assert exc_info.value.code == 0