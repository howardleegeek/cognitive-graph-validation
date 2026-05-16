#!/usr/bin/env python3
"""
Run script for H1.371 experiment
"""

import subprocess
import sys

def main():
    print("Running H1.371: Multi-step Coordinated Interactions Experiment")
    print("=" * 60)
    
    # Run the experiment
    result = subprocess.run([sys.executable, "experiment.py"], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode

if __name__ == "__main__":
    exit(main())