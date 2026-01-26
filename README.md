Credit Risk and Loan Default Prediction System
Machine learning system for credit risk assessment using German Credit and Loan Approval datasets. Implements multiple classification algorithms with comprehensive exploratory data analysis, feature engineering, and model comparison visualizations.
📊 Project Overview
This project analyzes credit risk using two distinct datasets:

German Credit Dataset: Assesses individual creditworthiness based on financial and personal attributes
Loan Approval Dataset: Predicts loan approval/rejection based on applicant demographics and financial history

The system trains and compares three machine learning models, performs extensive feature engineering, and generates detailed visualizations for model interpretation.
🎯 Features

Multiple ML Models: Logistic Regression, Random Forest, Gradient Boosting
Automated EDA: Comprehensive exploratory data analysis with 9+ visualization plots per dataset
Feature Engineering: Custom risk scores, income ratios, and composite indicators
Model Comparison: Side-by-side performance metrics and ROC curves
Prediction Analysis: Confusion matrices, probability distributions, and error analysis
Feature Importance: Visual ranking of most influential predictors
Cross-Validation: 5-fold stratified CV for robust performance estimates
Automated Reporting: JSON export of all results and conclusions

🚀 Getting Started
Prerequisites
bashPython 3.7+
Installation

Clone the repository:

bashgit clone https://github.com/yourusername/credit-risk-analysis.git
cd credit-risk-analysis

Install required packages:

bashpip install pandas numpy matplotlib seaborn scikit-learn
Usage
Run the complete analysis:
bashpython credit_risk_analysis.py
The script will:

Load or generate sample datasets
Perform exploratory data analysis
Train multiple ML models
Generate visualizations
Create a final report

📁 Project Structure
credit-risk-analysis/
│
├── credit_risk_analysis.py          # Main analysis script
├── README.md                         # This file
│
├── data/                             # (Optional) Place your datasets here
│   ├── credit_risk.csv
│   └── loan_train.csv
│
└── output/                           # Generated files
    ├── german_credit_eda.png
    ├── loan_approval_eda.png
    ├── german_model_comparison.png
    ├── loan_model_comparison.png
    ├── german_prediction_analysis.png
    ├── loan_prediction_analysis.png
    ├── dataset_comparison.png
    ├── german_feature_importance.png
    ├── loan_feature_importance.png
    ├── project_report.json
    ├── german_credit_model.pkl
    └── loan_approval_model.pkl
📊 Datasets
German Credit Dataset

Size: 1,000 samples
Features: 17 attributes including checking balance, credit history, loan purpose, employment duration
Target: Binary classification (default/no default)

Loan Approval Dataset

Size: 600+ samples
Features: 12 attributes including income, loan amount, credit history, property area
Target: Binary classification (approved/rejected)

🔬 Methodology
1. Data Preprocessing

Missing value imputation (median for numeric, mode for categorical)
Label encoding for categorical variables
Feature scaling using StandardScaler

2. Feature Engineering

German Credit: Risk scores for checking balance, savings, employment
Loan Approval: Total income, loan-to-income ratio, monthly EMI calculations
Composite risk indicators

3. Model Training
Three models trained with optimized hyperparameters:

Logistic Regression: L2 regularization, balanced class weights
Random Forest: 50-100 estimators, max depth constraints
Gradient Boosting: 50-100 estimators, learning rate optimization

4. Evaluation Metrics

Accuracy
Precision
Recall
F1-Score
ROC-AUC (primary metric)
5-fold cross-validation scores

📈 Results
German Credit Dataset

Best Model: Random Forest
AUC: ~0.75-0.80
Key Features: Checking balance, credit history, loan duration

Loan Approval Dataset

Best Model: Gradient Boosting / Random Forest
AUC: ~0.70-0.85
Key Features: Credit history, total income, loan amount

🖼️ Visualizations
The system generates 9 comprehensive visualization files:

EDA Plots: Distribution analysis, default rates by categories
Model Comparison: Bar charts comparing accuracy, AUC, F1-scores
ROC Curves: Multi-model performance comparison
Prediction Analysis: Confusion matrices, probability distributions
Feature Importance: Top 10 most influential features
Dataset Comparison: Cross-dataset model performance

🛠️ Configuration
Custom Dataset Usage
Place your CSV files in the project directory with names:

credit_risk.csv or german_credit.csv (German Credit data)
loan_train.csv or train.csv (Loan Approval data)

Model Hyperparameters
Modify in the train_german_models() and train_loan_models() methods:
pythonmodels = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,  # Adjust here
        max_depth=10,      # Adjust here
        random_state=42
    )
}
📝 Output Files

PNG Images: High-resolution (300 DPI) visualizations
JSON Report: Complete analysis summary with metrics
PKL Models: Serialized best models for deployment

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👥 Authors

Andy Zhu (AZ455)
Jacob Cheng (JC3045)

Course: CS 439: Introduction to Data Science
🙏 Acknowledgments

German Credit Dataset: UCI Machine Learning Repository
Loan Approval Dataset: Analytics Vidhya
scikit-learn documentation and community

📧 Contact
For questions or feedback, please open an issue on GitHub.
