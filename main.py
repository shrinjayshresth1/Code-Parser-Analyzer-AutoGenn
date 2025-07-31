#!/usr/bin/env python3
"""
Main entry point for the AutoGen + GenAI Enhanced Code Analyzer.
Run this script to start the interactive analyzer.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from code_analyzer.analyzer import main

if __name__ == "__main__":
    main()
