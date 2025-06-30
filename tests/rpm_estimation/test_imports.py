"""
Smoke tests to ensure all modules can be imported.
"""

import pytest


def test_all_imports():
    """Ensure all modules can be imported without errors."""
    # Main module
    import src.analysis.rpm as rpm_estimation
    
    # Sub-modules
    import src.analysis.rpm.io
    import src.analysis.rpm.preprocess
    import src.analysis.rpm.spectral
    import src.analysis.rpm.tracking
    import src.analysis.rpm.fusion
    import src.analysis.rpm.cli
    
    # Check version
    assert hasattr(rpm_estimation, '__version__')
    assert isinstance(rpm_estimation.__version__, str)


def test_main_exports():
    """Test that main module exports key components."""
    import src.analysis.rpm as rpm_estimation
    
    # Check key exports
    assert hasattr(rpm_estimation, 'RPMFrame')
    assert hasattr(rpm_estimation, 'cli_main')
    
    # Test RPMFrame is accessible
    from src.analysis.rpm import RPMFrame
    assert RPMFrame is not None


def test_cli_imports():
    """Test CLI module imports and parser creation."""
    from src.analysis.rpm.cli import create_parser, main
    
    # Test parser creation
    parser = create_parser()
    assert parser is not None
    
    # Test help doesn't crash
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(['--help'])
    assert exc_info.value.code == 0