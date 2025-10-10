# Retail-Sales-Forecasting-Pipeline
An automated time series retail sales forecasting model with an ETL pipeline

## Table of Contents
- [Overview](#overview)  
- [Business Problem](#business-problem)
- [Project Structure](#project-structure)
- [Installation](#installation)  
- [Usage](#usage)   
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [Technologies Used](#technologies-used)
- [Model Performance & Data Limitations](#model-performance--data-limitations) 
- [Dataset](#dataset)
- [Contact](#contact)
- [License](#license)

## Overview
This project implements a complete machine learning pipeline for retail sales forecasting, including:
- Automated ETL data processing
- Time-series feature engineering (lag features, rolling statistics, cyclical encoding)
- Multi-model training and comparison (XGBoost vs Random Forest)
- Comprehensive model evaluation and visualization

Key Features:
- Store-category level sales aggregation
- 37 engineered features from temporal, business, and external factors
- Time-based train/test splitting to prevent data leakage
- Production-ready modular architecture

## Business Problem
Accurate sales forecasting is critical for:
- Inventory Management: Preventing stockouts and overstock situations
- Resource Planning: Optimizing staff scheduling and warehouse operations
- Financial Planning: Improving revenue predictions and budget allocation
- Marketing Strategy: Timing promotions and campaigns effectively

## Project Structure
retail-sales-forecasting/  
├── data/  
│   ├── raw/                            
│   │   └── retail_store_inventory.csv   # Original dataset  
│   └── processed/                       # Generated datasets (not tracked)  
│       ├── retail_features_complete.csv  
│       ├── train_data.csv  
│       ├── test_data.csv  
│       └── feature_columns.json  
├── src/  
│   ├── __init__.py  
│   ├── data_loader.py                   # Data loading and aggregation  
│   ├── feature_engineering.py           # Feature creation functions  
│   ├── models.py                        # Model class definitions  
│   └── evaluation.py                    # Performance metrics  
├── scripts/  
│   ├── process_data.py                  # Stage 1: Data processing pipeline  
│   └── train_model.py                   # Stage 2: Model training pipeline  
├── notebooks/  
│   ├── 00_data_exploration.ipynb        # Exploratory data analysis  
│   └── model_comparison.ipynb           # Model evaluation and visualization  
├── trained_models/                      # Saved models (not tracked)  
│   ├── xgboost_model.pkl  
│   ├── randomforest_model.pkl  
│   └── feature_columns.json  
├── .gitignore  
├── config.py  
├── LICENSE.txt  
├── requirements.txt  
└── README.md  

## Installation
```bash
# Clone repository
git clone https://github.com/ElliottYankee/Retail-Sales-Forecasting-Pipeline.git
cd retail-sales-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Optional: Examine raw data
jupyter notebook notebooks/00_dataset_overview.ipynb

# Stage 1: Process raw data
python scripts/process_data.py

# Stage 2: Train model
python scripts/train_model.py

# Optional: Compare model outputs
jupyter notebook notebooks/model_comparison.ipynb
```

## What This Project Demonstrates  
While the performance metrics are inflated, the project successfully showcases:  
- End-to-end ML pipeline architecture  
- Proper time-series data handling (no data leakage)  
- Feature engineering for temporal forecasting  
- Model comparison and evaluation frameworks  
- Production-ready code structure  

## Technologies Used
Core:  
- Python 3.12  
- pandas, NumPy (data processing)  
- scikit-learn (Random Forest, metrics)  
- XGBoost (gradient boosting)  

Visualization:
- Matplotlib, Seaborn  
- Jupyter Notebook  

Infrastructure:
- joblib (model persistence)
- Git (version control)  

## Model Performance & Data Limitations
### Current Results

Both models achieve exceptionally high performance metrics:
- **XGBoost**: MAPE 2.11%, R² 0.999
- **RandomForest**: MAPE 1.54%, R² 0.999

### Important Caveat
**These results are unrealistically high due to synthetic data limitations.**
Real-world retail forecasting typically achieves:
- MAPE: 10-20% (acceptable performance)
- R²: 0.70-0.85 (good explanatory power)

The synthetic dataset used in this project exhibits artificially predictable patterns, lacking the noise, irregularities, and complexity typically found in actual retail operations. This enables models to achieve near-perfect predictions that would be impossible with real data.

## Dataset
This project uses the [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) from Kaggle.

**Dataset characteristics:**
- 73,100+ daily records across 5 stores and 20 products
- 5 product categories: Electronics, Clothing, Groceries, Furniture, Toys
- Features include sales, inventory, pricing, weather, and promotional data
- Date range: January 2022 - January 2024
  Data Processing:

**Data Processing:**  
- Aggregated from product-level to store-category level (73,100 to 46,947 records)  
- Removed problematic "Demand Forecast" column (0.997 correlation with target)  
- Fixed incorrect seasonality assignments  
- Applied data quality validations  

**Note:** This is synthetic data created for educational purposes. See [Model Performance & Data Limitations](#model-performance--data-limitations) for details on how this affects results.

## Contact
**Elliott Yankelevich**
- [Email](elliottyankelevich@gmail.com)
- [LinkedIn](www.linkedin.com/in/elliott-yankelevich)
- [GitHub](https://github.com/ElliottYankee)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
