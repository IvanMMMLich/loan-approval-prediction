# 🌊 Flood Risk Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Project Overview

Machine Learning project for predicting flood probability based on 20 risk factors using regression models.

- **Task Type:** Regression  
- **Target Variable:** FloodProbability (0.0 - 1.0)  
- **Features:** 20 risk factors (scale 1-10)
- **Evaluation Metrics:** RMSE, MAE, R²

## 🎯 Business Goal

Develop a model to predict flood risk probability to help:
- Emergency services prepare resources
- Urban planners identify high-risk areas  
- Insurance companies assess risks
- Government agencies allocate prevention budgets

## 📊 Dataset

- **Training set:** Unknown size (will update after EDA)
- **Test set:** Unknown size (will update after EDA)
- **Features:** 20 numerical features representing various risk factors

## 🚀 Quick Start
```bash
# 1. Clone repository
git clone https://github.com/IvanMMMLich/flood-risk-prediction.git
cd flood-risk-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add data files to data/raw/
# Download train.csv, test.csv, sample_submission.csv

# 5. Run analysis (coming soon)
python scripts/run_eda.py
```

## 📁 Project Structure
```
flood-risk-prediction/
├── data/              # Data files (not tracked in Git)
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
│   ├── 01_eda/       # Exploratory Data Analysis
│   ├── 02_preprocessing/  # Feature Engineering
│   ├── 03_modeling/   # Model Training
│   └── 04_evaluation/ # Model Evaluation
├── models/            # Saved models
└── results/           # Outputs and reports
```

## 🔬 Methodology

1. **EDA** - Understanding data patterns
2. **Feature Engineering** - Creating risk indices
3. **Modeling** - Testing multiple regression algorithms
4. **Evaluation** - Cross-validation and metrics analysis

## 📈 Current Status

- [x] Repository setup
- [ ] Data loading
- [ ] EDA
- [ ] Feature Engineering  
- [ ] Model Training
- [ ] Submission

## 👤 Author

**Ivan Sytsev**
- GitHub: [@IvanMMMLich](https://github.com/IvanMMMLich)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
