"""Smoke tests for the Mortgage Credit Risk package."""

from mortgage_credit_risk import __version__


def test_package_imports() -> None:
    """Verify the package exposes a non-empty version string."""
    assert isinstance(__version__, str)
    assert __version__
