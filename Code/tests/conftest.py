"""Shared set-up of the tests.

Run all tests from the repository root with:  python -m pytest Code/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The code folders are not Python packages, so their modules are imported by path.
CODE_DIR = Path(__file__).resolve().parents[1]
for folder in ("Preprocessing", "Classification Algorithms", "Postprocessing"):
    sys.path.insert(0, str(CODE_DIR / folder))


@pytest.fixture(scope="session")
def features():
    """The feature table of all 338 windows, computed once for all tests."""
    from extract_features import extract_features
    return extract_features(verbose=False)


@pytest.fixture
def features_file(features, tmp_path):
    """The feature table written to a temporary CSV file."""
    path = tmp_path / "features.csv"
    features.to_csv(path, index=False)
    return path
