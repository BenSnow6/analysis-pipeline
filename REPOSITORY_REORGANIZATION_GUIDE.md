# Repository Reorganization Guide

## Current Status
- **Current Branch**: `data_analysis`
- **Uncommitted Changes**: 768 files (mostly CSV data files)
- **Python Files**: 107 total
- **Internal Import Dependencies**: 76 files

## Pre-Migration Checklist

### 1. Git Preparation
```bash
# First, commit or stash current changes
git add -A
git commit -m "WIP: Pre-reorganization checkpoint"

# Create new branch for reorganization
git checkout -b repo-reorganization

# Ensure you have a backup
git branch backup-pre-reorg
```

### 2. Critical Issues to Address

#### Hardcoded Paths (4 files)
1. `hovercraft_data_analysis/dashboard_app/config.py` - Uses `../../02_Evaluation_Experiments`
2. `hovercraft_data_analysis/orientation_analysis/analyze_static_gyro_simple.py` - Hardcoded CSV paths
3. `hovercraft_data_analysis/orientation_analysis/check_gyro_units.py` - Hardcoded CSV paths
4. `thesis_analysis/scripts/simulator_validation.py` - Placeholder absolute paths

#### sys.path Manipulations (32 files)
- Most files in `code/rpm_estimation/`
- Many files in `hovercraft_data_analysis/orientation_analysis/`

## Target Structure
```
analysis-pipeline/
├── /code              # All Python code
├── /thesis            # Thesis chapters and LaTeX
├── /docs              # Documentation, READMEs, reports
└── /data              # All experimental data
```

## Migration Plan

### Phase 1: Prepare Structure
```bash
# Create target directories
mkdir -p code thesis docs data

# Create proper Python package structure
mkdir -p code/src code/tests code/scripts code/notebooks
touch code/__init__.py code/src/__init__.py
```

### Phase 2: Move Data Files
```bash
# Move experimental data (choose ONE organization)
# Option A: Keep experiment-based structure
mv 02_Evaluation_Experiments/* data/
rm -rf all_expts  # Remove duplicate organization

# Option B: Keep time-based structure
mv all_expts/* data/
rm -rf 02_Evaluation_Experiments
```

### Phase 3: Move Code Files

#### Core Modules
```bash
# Move main source code
mv src/* code/src/
mv hovercraft_data_analysis code/src/
mv code/rpm_estimation code/src/

# Move root Python scripts
mv *.py code/scripts/

# Move notebooks
mv notebooks/* code/notebooks/
mv *.ipynb code/notebooks/
```

### Phase 4: Move Documentation
```bash
# Move all documentation
mv README.md docs/
mv MEGA_MARKDOWN.md docs/
mv _tree.md docs/
mv codebase_analysis docs/
mv timestamp_analysis_results docs/
mv results docs/

# Move thesis content
mv notes/* thesis/
mv thesis_analysis/* thesis/
mv hovercraft_data_analysis/plans/thesis_plan.md thesis/
```

### Phase 5: Update Imports

#### 1. Create Path Configuration
Create `code/config/paths.py`:
```python
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
THESIS_DIR = PROJECT_ROOT / "thesis"

# Data paths
EXPERIMENTS_DIR = DATA_DIR / "experiments"
RESULTS_DIR = DOCS_DIR / "results"
```

#### 2. Update Import Statements
Replace all relative imports with absolute imports from the new structure:

**Before:**
```python
from src.data_processing import process_data
sys.path.append('..')
from frame_definitions import FRAMES
```

**After:**
```python
from code.src.data_processing import process_data
from code.src.frame_definitions import FRAMES
```

#### 3. Update File Paths
Replace hardcoded paths with configuration-based paths:

**Before:**
```python
df = pd.read_csv("../../02_Evaluation_Experiments/1a_1/data.csv")
```

**After:**
```python
from code.config.paths import EXPERIMENTS_DIR
df = pd.read_csv(EXPERIMENTS_DIR / "1a_1/data.csv")
```

### Phase 6: Update Configuration Files

1. Update `.gitignore`:
```
/data/
/venv/
__pycache__/
*.pyc
.pytest_cache/
```

2. Create `setup.py` in root:
```python
from setuptools import setup, find_packages

setup(
    name="hovercraft-analysis",
    version="0.1.0",
    packages=find_packages(where="code"),
    package_dir={"": "code"},
    install_requires=[
        # Add your requirements here
    ],
)
```

3. Update `requirements.txt` with all dependencies

### Phase 7: Testing

1. Test critical scripts:
```bash
# Test imports
python -c "from code.src.hovercraft_data_analysis.alignment import align"

# Run tests
pytest code/tests/

# Test key functionality
python code/scripts/dashboard_app.py
```

2. Verify data paths work correctly
3. Check all notebooks still run

## Automated Migration Script

Save this as `migrate_repo.py`:
```python
#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

def update_imports_in_file(filepath):
    """Update import statements in a Python file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Update imports
    replacements = [
        (r'from src\.', 'from code.src.'),
        (r'import src\.', 'import code.src.'),
        (r'from hovercraft_data_analysis\.', 'from code.src.hovercraft_data_analysis.'),
        (r'sys\.path\.append\([^)]+\)', '# sys.path.append removed'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)

# Run migration
if __name__ == "__main__":
    print("Starting repository reorganization...")
    # Add migration logic here
```

## Post-Migration Checklist

- [ ] All tests pass
- [ ] Dashboard app runs
- [ ] Key notebooks execute without errors
- [ ] No broken imports
- [ ] Data paths resolve correctly
- [ ] Documentation is accessible
- [ ] Git history is preserved
- [ ] Create new README.md in root explaining structure

## Rollback Plan

If something goes wrong:
```bash
git checkout backup-pre-reorg
git branch -D repo-reorganization
```

## Notes for AI Assistant

When executing this migration:
1. Always create backups first
2. Test each phase before moving to the next
3. Update imports incrementally and test
4. Keep a log of all changes made
5. Don't delete anything until confirmed working