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

All tasks use rolling-origin evaluation with multiple cutoffs to ensure robust performance assessment.
The following table summarizes the available tasks in the benchmark:

| dataset           | hierarchy_level | frequency_level | forecasting_horizon | number_of_rows | number_of_timestamps |
| ----------------- | --------------- | --------------- | ------------------- | -------------- | -------------------- |
| m5                | product         | weekly          | 4                   | 845444         | 277                  |
| m5                | product         | monthly         | 3                   | 197270         | 64                   |
| m5                | store           | daily           | 7                   | 19410          | 1941                 |
| favorita          | product         | weekly          | 4                   | 332002         | 241                  |
| favorita          | product         | monthly         | 3                   | 77467          | 56                   |
| favorita          | store           | daily           | 7                   | 31094          | 1688                 |
| favorita          | store           | weekly          | 4                   | 4442           | 241                  |
| rohlik            | product/store   | weekly          | 4                   | 708704         | 200                  |
| rohlik            | product         | daily           | 7                   | 708704         | 1402                 |
| rohlik            | product         | weekly          | 4                   | 101243         | 200                  |
| rossmann          | product/store   | weekly          | 4                   | 290631         | 134                  |
| rossmann          | store           | weekly          | 4                   | 145315         | 134                  |
| bakery            | product/store   | daily           | 7                   | 127575         | 1215                 |
| bakery            | product/store   | weekly          | 4                   | 18225          | 173                  |
| bakery            | product         | daily           | 7                   | 3645           | 1215                 |
| bakery            | store           | daily           | 7                   | 42525          | 1215                 |
| bakery            | store           | weekly          | 4                   | 6075           | 173                  |
| yaz               | product         | daily           | 7                   | 5355           | 765                  |
| pharmacy          | product         | weekly          | 4                   | 54621          | 119                  |
| pharmacy2         | product/store   | daily           | 7                   | 279330         | 684                  |
| pharmacy2         | product/store   | weekly          | 4                   | 39904          | 97                   |
| freshretail50k    | product         | daily           | 7                   | 5011           | 90                   |
| freshretail50k    | store           | daily           | 7                   | 5202           | 90                   |
| hoteldemand       | product/store   | daily           | 7                   | 46508          | 3227                 |
| hoteldemand       | product/store   | weekly          | 4                   | 6644           | 461                  |
| hoteldemand       | product         | daily           | 7                   | 5813           | 3227                 |
| hoteldemand       | store           | daily           | 7                   | 23254          | 3227                 |
| hoteldemand       | store           | weekly          | 4                   | 3322           | 461                  |
| onlineretail      | product         | weekly          | 4                   | 217454         | 53                   |
| onlineretail2     | product         | weekly          | 4                   | 269021         | 105                  |
| australianretail  | product/store   | monthly         | 3                   | 64532          | 441                  |
| australianretail  | product         | monthly         | 3                   | 8066           | 441                  |
| australianretail  | store           | monthly         | 3                   | 3226           | 441                  |
| kaggledemand      | product/store   | weekly          | 4                   | 150150         | 130                  |
| kaggledemand      | store           | weekly          | 4                   | 5362           | 130                  |
| productdemand     | product/store   | weekly          | 4                   | 689684         | 313                  |
| productdemand     | product/store   | monthly         | 3                   | 160926         | 73                   |
| productdemand     | product         | weekly          | 4                   | 172421         | 313                  |
| productdemand     | product         | monthly         | 3                   | 40231          | 73                   |
| vn1               | product         | weekly          | 4                   | 7801           | 170                  |
| kagglewalmart     | store           | weekly          | 4                   | 421570         | 143                  |
| hierarchicalsales | product         | daily           | 7                   | 212164         | 1825                 |
| hierarchicalsales | product         | weekly          | 4                   | 30309          | 260                  |
| hierarchicalsales | product         | monthly         | 3                   | 7072           | 60                   |
| carparts          | product         | monthly         | 3                   | 136374         | 51                   |
| fossil            | product         | monthly         | 3                   | 44907          | 70                   |


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

