#!/usr/bin/env python3
"""
Run script for H1.370: Multi-Object Interaction Requirement for Cognitive Graph Advantage
"""

import subprocess
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment import main

if __name__ == "__main__":
    print("Running H1.370: Multi-Object Interaction Requirement for Cognitive Graph Advantage")
    print("=" * 80)
    main()