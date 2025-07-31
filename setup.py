#!/usr/bin/env python3
"""
Setup script for AutoGen + GenAI Enhanced Code Analyzer
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="autogen-code-analyzer",
    version="1.0.0",
    author="GitHub Copilot",
    description="AutoGen + GenAI Enhanced Code Parser and Analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devsinghsolanki/Code-Parser-Analyzer-AutoGenn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "code-analyzer=code_analyzer.analyzer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/.env.template"],
    },
)
