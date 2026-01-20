import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f"✅ Created: {path}")

# ==========================================
# FILE CONTENTS
# ==========================================

README_CONTENT = """
# Quant ML: Sentiment-Augmented Price Prediction

![CI Status](https://github.com/jmiaie/quant-ml-project/actions/workflows/main.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📌 Overview
This project implements a rigorous Machine Learning pipeline for predicting short-term asset price movements. It emphasizes **Financial Rigor** (Walk-Forward Validation) and **Interpretability** (SHAP).

## 🏗 Architecture
* **`src/validation.py`**: Custom Walk-Forward Validator to prevent look-ahead bias.
* **`src/features.py`**: Advanced technical indicators (RSI, Bollinger Z-Scores).
* **`src/sentiment.py`**: NLP engine using FinBERT.

## 🚀 Quick Start
```bash
# Docker
docker build -t quant-ml .
docker run quant-ml python backtest.py
