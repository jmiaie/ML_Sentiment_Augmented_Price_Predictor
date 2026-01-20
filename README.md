# ML_Sentiment_Augmented_Price_Predictor
Machine Learning: Sentiment-Augmented Price Predicting Algorithm
# Quant ML: Sentiment-Augmented Price Prediction

![CI Status](https://github.com/jmiaie/quant-ml-project/actions/workflows/main.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## 📌 Overview

**Sentiment-Augmented Alpha** is a rigorous Machine Learning pipeline designed to predict short-term asset price movements. Unlike standard data science projects, this implementation prioritizes **Financial Rigor** and **Interpretability** over raw theoretical accuracy.

This project simulates a professional Quantitative Research environment by addressing common pitfalls:
1.  **Look-Ahead Bias:** Solved via a custom Walk-Forward Validation framework.
2.  **Black Box Risk:** Mitigated using SHAP (SHapley Additive exPlanations) for feature transparency.
3.  **Reproducibility:** Fully containerized with Docker and automated via GitHub Actions.

## 🏗 Architecture

The system follows a modular data engineering pipeline, transforming raw unstructured data into stationary signals.

```mermaid
graph LR
    A[Raw Market Data] --> C{Feature Engineering}
    B[Social Sentiment] --> C
    C --> D[Stationary Features<br/>(RSI, Z-Scores)]
    D --> E{Walk-Forward<br/>Validator}
    E --> F[XGBoost Model]
    F --> G[Performance Metrics<br/>(Sharpe, Precision)]
    F --> H[SHAP Interpretation]
