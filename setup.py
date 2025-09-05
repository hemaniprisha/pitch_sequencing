"""
Setup script for MLB Pitch Sequencing Optimization package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mlb-pitch-sequencing",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced baseball analytics for MLB pitch sequencing optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/mlb-pitch-sequencing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Other/Nonlisted Topic",  # Sports analytics
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0"
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.15.0",
            "ipywidgets>=7.7.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "mlb-pitch-sequencing=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)