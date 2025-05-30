# Benchmarking for Demand Forecasting in Supply Chain Management

**Tagline:**  
A standardized framework to benchmark and compare state-of-the-art demand-forecasting models on real-world demand data.

---

## 📖 Overview

Many existing time-series benchmarks rely on tasks (e.g., stock-price prediction) that are only tangentially related to supply-chain needs. **Benchmarking for Demand Forecasting in Supply Chain Management** fills this gap by providing:

- **Real-world datasets** 
- **A unified evaluation pipeline** 
- **Out-of-the-box metrics** 

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.11 
- `pip`  

### Installation

1. Clone the repo:  
2. Install the requirements:  
```bash
pip install -r requirements.txt
```

## 🔧 Configuring

All pipeline settings live in config/. You’ll find:
config/config.example.yaml — a template for your own config/config.yaml
**DO NOT commit** your own config/config.yaml; it’s ignored by .gitignore

## 🎯 Usage

You have two ways to run the benchmarking pipeline—either directly on your local machine or inside Docker.

### 🏃‍♂️ Local

1. Make sure you’ve installed the requirements and set up your config:
```bash
pip install -r requirements.txt
```
2. Edit the public config file `config/public/config.yaml` to your liking. If you want to use weights and biases (wandb) for logging, you can write your credentials in the private config file `config/private/config.yaml` file and set the `use_wandb` flag in the public config file to `True`. The private config file is not tracked by git, so you can safely add your credentials there.

3. Run the pipeline with:
```bash
python -m src.main -c config/public/config.yaml -s config/private/config.yaml
```

### 🐳 Docker

Build the container (only needs to be done once):
```bash
docker build -t forecast-bench .
```
Run with a bind mount to persist Feather outputs into your local data/ folder:
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  forecast-bench \
  -c config/public/config.yaml -s config/private/config.yaml
```
This maps your host’s ./data directory into the container’s /app/data, so any files written there (e.g. *.feather) appear locally.

## 📄 License & Authors

**License:** MIT
**Authors:** Moritz Beck, Anh-Duy Pham

