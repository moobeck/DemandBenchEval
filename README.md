# DemandBench

**DemandBench** is introduced as a data- and protocol-first benchmark for demand forecasting. At its core is a unified panel schema with identifiers, timestamps, targets, covariates, and optional hierarchical structure, complemented by machine-readable metadata and dataset cards that document provenance, curation steps, licensing, and usage caveats. Tasks are specified by combining dataset, hierarchical level, sampling frequency, and forecasting horizon so as to mirror the settings in which practitioners make inventory, pricing, and staffing decisions. The evaluation protocol adopts rolling-origin assessment and reports both point and probabilistic accuracy using scale-robust criteria suitable for sparse and intermittent series.

## ✨ Features

- **Comprehensive Model Support**: Includes statistical models (ARIMA, ETS), deep learning models (LSTM, Transformer), and foundation models (Chronos, Moirai, TabPFN).
- **Real-World Datasets**: Benchmark on datasets including M5, Favorita, ROHLIK, Rossmann, Bakery, Yaz, Pharmacy, Hotel Demand, Online Retail, Fresh Retail 50K, Hierarchical Sales, Australian Retail, Car Parts, Kaggle Demand, Product Demand, VN1, Kaggle Retail, Kaggle Walmart, and Fossil.
- **Flexible Evaluation**: Supports multiple metrics including MASE, RMSE, MAE, and probabilistic metrics like SMQL.
- **Easy Configuration**: Modular YAML-based configuration for datasets, models, and evaluation settings.
- **Docker Support**: Run the entire pipeline in a containerized environment.
- **Experiment Tracking**: Optional integration with Weights & Biases (Wandb) for logging and visualization.

## 🤖 Supported Models

The framework supports a wide range of forecasting models across different categories:

### Statistical Models
- ARIMA, THETA, ETS, CES

### Deep Learning Models
- MLP, LSTM, GRU, TCN, Transformer, TFT, PatchTST, XLSTM
- TimesNet, FEDformer, TiDE, N-HiTS, DeepAR, N-BEATS, BITCN

### Foundation Models
- Chronos, Moirai, TabPFN

## 📊 Datasets

The benchmark includes a comprehensive collection of real-world demand forecasting datasets:

- M5
- Favorita
- ROHLIK
- Rossmann
- Bakery
- Yaz
- Pharmacy
- Pharmacy2
- Hotel Demand
- Online Retail
- Online Retail 2
- Fresh Retail 50K
- Hierarchical Sales
- Australian Retail
- Car Parts
- Kaggle Demand
- Product Demand
- VN1
- Kaggle Retail
- Kaggle Walmart
- Fossil

## 🎯 Tasks

Tasks are defined as combinations of dataset, hierarchical level, sampling frequency, and forecasting horizon. The registry includes numerous predefined tasks that mirror real-world decision-making scenarios for inventory, pricing, and staffing. Examples include:

- Product/Store-level forecasting at daily, weekly, or monthly frequencies
- Product-level aggregations at all frequencies
- Store-level aggregations at all frequencies
- Various forecasting horizons (e.g., 3 months, 4 weeks, 7 days)

All tasks use rolling-origin evaluation for robust assessment.

---

## 📈 Metrics

The framework provides comprehensive evaluation metrics for demand forecasting:

### Point Forecast Metrics
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)  
- **RMSE** (Root Mean Squared Error)

### Scale-Invariant Metrics
- **MASE** (Mean Absolute Scaled Error)
- **MSSE** (Mean Squared Scaled Error)

### Probabilistic Metrics
- **SMQL** (Scaled Mean Quantile Loss) - for quantile-based evaluation

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

All pipeline settings are organized in the `config/` directory with separate files for different aspects:

- `config/public/system.yaml` - System settings (GPU, random seed)
- `config/public/filepaths.yaml` - Data and output file paths
- `config/public/forecast.yaml` - Model configurations
- `config/public/metrics.yaml` - Evaluation metrics
- `config/public/task.yaml` - Dataset and task definitions

Private configurations (e.g., Wandb credentials) go in `config/private/`. **DO NOT commit** your private config files; they are ignored by .gitignore.

---

## 🎯 Usage

You have two ways to run the benchmarking pipeline—either directly on your local machine or inside Docker.

### 🏃‍♂️ Local

1. Make sure you’ve installed the requirements and set up your config:
```bash
pip install -r requirements.txt
```
2. Edit the public and private config YAML files as needed.

3. Run the pipeline with:
```bash
python -m src.main --config-dir config
```

### 🐳 Docker

Build the container (only needs to be done once):
```bash
docker build --build-arg GITHUB_TOKEN=your_github_token_here -t demandbench .
```
Run with a bind mount to persist Feather outputs into your local data/ folder:
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  demandbench \
 --config-dir config
```
This maps your host’s ./data directory into the container’s /app/data, so any files written there appears locally.

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss the proposed changes.

## 📄 License & Authors

**License:** MIT
**Authors:** Moritz Beck, Anh-Duy Pham

