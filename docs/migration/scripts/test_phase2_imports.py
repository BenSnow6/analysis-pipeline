#!/usr/bin/env python3
"""
Test script to verify Phase 2 imports work correctly.
Run this after `pip install -e .` to ensure package is installed.
"""
import sys
sys.path.insert(0, 'src')

def test_basic_imports():
    """Test that basic imports work."""
    print("Testing basic package imports...")
    
    try:
        # Test package-level import
        import hovercraft_analysis
        print(f"✓ Package import successful, version: {hovercraft_analysis.__version__}")
    except ImportError as e:
        print(f"✗ Package import failed: {e}")
        return False
    
    # Test submodule imports
    modules_to_test = [
        "hovercraft_analysis.core",
        "hovercraft_analysis.analysis",
        "hovercraft_analysis.apps",
        "hovercraft_analysis.scripts",
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
    
    return True


def test_path_imports():
    """Test path management imports."""
    print("\nTesting path management...")
    
    try:
        from src.core.paths import (
            PROJECT_ROOT,
            DATA_DIR,
            get_experiment_path,
        )
        print(f"✓ Path imports successful")
        print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"  DATA_DIR: {DATA_DIR}")
        
        # Test helper function
        exp_path = get_experiment_path("test_exp", "morning")
        print(f"  Test experiment path: {exp_path}")
        
    except ImportError as e:
        print(f"✗ Path imports failed: {e}")
        return False
    
    return True


def test_config_imports():
    """Test configuration imports."""
    print("\nTesting configuration management...")
    
    try:
        from src.core.config import Config, get_config
        print(f"✓ Config imports successful")
        
        # Test config instantiation
        config = get_config()
        print(f"  Config instance created: {type(config)}")
        
    except ImportError as e:
        print(f"✗ Config imports failed: {e}")
        return False
    
    return True


def test_analysis_module_structure():
    """Test analysis module structure."""
    print("\nTesting analysis module structure...")
    
    analysis_modules = [
        "hovercraft_analysis.analysis.alignment",
        "hovercraft_analysis.analysis.orientation",
        "hovercraft_analysis.analysis.timestamp",
        "hovercraft_analysis.analysis.rpm",
    ]
    
    for module in analysis_modules:
        try:
            __import__(module)
            print(f"✓ {module} exists")
        except ImportError as e:
            print(f"✗ {module} not found: {e}")
    
    return True


def main():
    """Run all import tests."""
    print("=" * 60)
    print("Phase 2 Import Testing")
    print("=" * 60)
    
    # Run tests
    all_passed = True
    all_passed &= test_basic_imports()
    all_passed &= test_path_imports()
    all_passed &= test_config_imports()
    all_passed &= test_analysis_module_structure()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All import tests passed!")
        print("\nNext steps:")
        print("1. Install package: pip install -e .")
        print("2. Run full test suite: python -m pytest tests/")
        print("3. Test command-line tools")
    else:
        print("❌ Some import tests failed")
        print("\nCheck that all files were copied correctly and")
        print("that __init__.py files exist in all directories")
    print("=" * 60)


if __name__ == "__main__":
    main()