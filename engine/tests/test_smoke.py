"""Smoke test: package imports and toolchain wiring."""

import superforecaster


def test_package_imports() -> None:
    assert superforecaster.__version__
