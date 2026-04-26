from pathlib import Path


def create_file(path: str, content: str) -> None:
    """Create a UTF-8 text file and ensure its parent directories exist."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.strip() + "\n", encoding="utf-8")
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
```
"""


def main() -> None:
    create_file("README.md", README_CONTENT)


if __name__ == "__main__":
    main()
