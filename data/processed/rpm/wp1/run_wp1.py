#!/usr/bin/env python3
"""
Sanity check script for WP1 processing
"""

import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.rpm.cli import main

if __name__ == "__main__":
    sys.exit(main())