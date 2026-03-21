"""
CET-Epi: Causal Emergence Theory for Epidemics
Setup configuration
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
req_path = Path(__file__).parent / "requirements.txt"
requirements = []
if req_path.exists():
    requirements = [line.strip() for line in req_path.read_text(encoding="utf-8").splitlines() 
                   if line.strip() and not line.startswith("#")]

setup(
    name="cet-epi",
    version="0.1.0",
    description="Causal Emergence Theory for Epidemics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dhruv Jyoti Das, Raghav Sharma",
    author_email="dd4708@srmist.edu.in, rs2701@srmist.edu.in",
    url="https://github.com/yourusername/cet-epi",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=8.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="epidemic-forecasting causal-emergence graph-neural-networks",
    entry_points={
        "console_scripts": [
            "cet-epi-train=src.training.trainer:main",
        ],
    },
)
