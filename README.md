# Retail-Sales-Forecasting-Pipeline
An automated time series retail sales forecasting model with an ETL pipeline

## Table of Contents
[Overview](#overview)  
[Business Problem](#business-problem)  
[Future Features](#future-features)  
[Dataset](#dataset)  
[Installation](#installation)  
[Usage](#usage)  
[Model Performance & Data Limitations](#model-performance--data-limitations)  
[What This Project Demonstrates](#what-this-project-demonstrates)
[Dataset](#dataset)

## Overview
An end-to-end machine learning system that predicts retail sales using time series analysis. This project demonstrates the complete ML lifecycle from data ingestion to model deployment, featuring automated ETL pipelines, multiple forecasting models, and a production-ready API.

## Business Problem
Accurate sales forecasting is critical for:
- Inventory Management: Preventing stockouts and overstock situations
- Resource Planning: Optimizing staff scheduling and warehouse operations
- Financial Planning: Improving revenue predictions and budget allocation
- Marketing Strategy: Timing promotions and campaigns effectively

## Future Features
- Automated ETL Pipeline: Processes raw sales data with validation and error handling
- Feature Engineering: Creates time-based, lag, and trend features for improved predictions
- Multi-Model Framework: Compares Linear Regression, Random Forest, and XGBoost models
- REST API: Provides real-time predictions via FastAPI
- Containerization: Docker support for easy deployment
- Monitoring: Tracks model performance and data quality over time
- Comprehensive Testing: Unit tests for all pipeline components

## Dataset
This project will use retail sales data featuring:
- Time Series Data: Daily sales records over multiple years
- Product Categories: Sales across different product lines
- Store Information: Multiple store locations and characteristics
- External Factors: Holidays, promotions, and seasonal events

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd retail-sales-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Stage 1: Process raw data
python scripts/process_data.py

# Stage 2: Train model
python scripts/train_model.py
```

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

The synthetic dataset used in this project exhibits artificially predictable patterns without the noise, irregularities, and complexity present in actual retail operations. This allows models to achieve near-perfect predictions that are impossible with real data.

## What This Project Demonstrates  
While the performance metrics are inflated, the project successfully showcases:
- End-to-end ML pipeline architecture
- Proper time-series data handling (no data leakage)
- Feature engineering for temporal forecasting
- Model comparison and evaluation frameworks
- Production-ready code structure

## Dataset

This project uses the [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) from Kaggle.

**Dataset characteristics:**
- 73,100+ daily records across 5 stores and 20 products
- 5 product categories: Electronics, Clothing, Groceries, Furniture, Toys
- Features include sales, inventory, pricing, weather, and promotional data
- Date range: January 2022 - January 2024

**Note:** This is synthetic data created for educational purposes. See [Model Performance & Data Limitations](#model-performance--data-limitations) for details on how this affects results.
