# CET-Epi: Causal Emergence Theory for Epidemics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![ROCm](https://img.shields.io/badge/ROCm-MI300X-d90000.svg)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)

**Causal Emergence Theory for Epidemics (CET-Epi)** is a novel deep learning framework that learns optimal multi-scale representations for epidemic forecasting through the lens of **Causal Emergence Theory**. Unlike traditional multi-scale models that use fixed geographic hierarchies, CET-Epi learns data-driven coarse-graining that maximizes **Effective Information (EI)** — creating macro-scale representations with greater causal power than micro-scale inputs.

## 🎯 Key Innovation

**Causal Emergence Operator (CEO)**: A differentiable module that learns to aggregate micro-scale units (counties) into macro-scale regions such that:
- EI_macro > EI_micro (emergence condition)
- Cross-scale intervention propagation is optimized
- Forecasting accuracy improves at both scales

## 🏗️ Architecture
┌─────────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Micro-Scale    │────▶│    CEO      │────▶│  Macro-Scale    │
│  (DCRNN)        │     │ (Learned    │     │  (DCRNN)        │
│  N counties     │     │  Coarse-    │     │  K regions      │
│                 │     │  graining)  │     │                 │
└─────────────────┘     └─────────────┘     └─────────────────┘
│                       │                       │
└───────────────────────┼───────────────────────┘
▼
┌─────────────────────┐
│  Cross-Scale        │
│  Attention         │
└─────────────────────┘
│
▼
┌─────────────────────┐
│  Scale-Aware        │
│  Predictor         │
└─────────────────────┘


## 🚀 Quick Start

### Prerequisites

- AMD MI300X GPU (or CUDA-compatible GPU with 24GB+ VRAM)
- Python 3.9+
- ROCm 5.7+ (for MI300X)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cet-epi.git
cd cet-epi

# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .