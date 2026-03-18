from setuptools import setup, find_packages

setup(
    name="cet-epi",
    version="0.1.0",
    description="Causal Emergence Theory for Epidemics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torch-geometric>=2.4.0",
        "torch-geometric-temporal>=0.54.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "networkx>=3.0",
        "pyyaml>=6.0",
    ],
)
