"""
Setup script for MAPSO - Multi-Agent Production Scheduling Optimizer

This file provides backwards compatibility for older pip versions.
For modern installations, pyproject.toml is preferred.
"""

from setuptools import setup, find_packages

setup(
    name="mapso",
    packages=find_packages(include=["mapso", "mapso.*", "dashboard", "dashboard.*"]),
    include_package_data=True,
)
