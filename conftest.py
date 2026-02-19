"""
conftest.py
-----------
Pytest configuration.  Adds the project root to sys.path so that
``import src.*`` works without installing the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
