# Credit Risk and Loan Default Prediction System

A machine learning project for predicting credit risk using the German Credit and Loan Approval datasets. The project compares multiple classification models, performs exploratory data analysis (EDA), applies feature engineering, and evaluates model performance.

## Features

- Logistic Regression, Random Forest, and Gradient Boosting
- Exploratory data analysis and visualizations
- Feature engineering
- Model comparison using Accuracy, Precision, Recall, F1 Score, and ROC-AUC
- Feature importance analysis
- 5-fold cross validation
- JSON report generation
- Saved trained models

## Installation

```bash
git clone https://github.com/yourusername/credit-risk-analysis.git
cd credit-risk-analysis
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

```bash
python credit_risk_analysis.py
```

The script will:
- Load the datasets
- Perform EDA
- Train and evaluate models
- Generate visualizations
- Save the best models and report

## Project Structure

```
credit-risk-analysis/
├── credit_risk_analysis.py
├── README.md
├── data/
└── output/
```

## Datasets

**German Credit Dataset**
- 1,000 samples
- 17 features
- Predicts loan default

**Loan Approval Dataset**
- 600+ samples
- 12 features
- Predicts loan approval

## Models

- Logistic Regression
- Random Forest
- Gradient Boosting

## Outputs

- EDA visualizations
- Model comparison plots
- ROC curves
- Confusion matrices
- Feature importance plots
- JSON report
- Trained model files (.pkl)

## Authors

- Andy Zhu (AZ455)
- Jacob Cheng (JC3045)

**Course:** CS 439: Introduction to Data Science

## License

MIT License
