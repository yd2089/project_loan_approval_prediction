# Loan Approval Prediction Model

## Project Overview

This project builds a predictive model to classify whether a loan application will be approved or not based on the "Loan Approval Classification Data" dataset from Kaggle. The model uses binomial regression (logistic regression) to predict binary outcomes: approval (1) or rejection (0).

## Dataset Information

- **Source**: [Kaggle - Loan Approval Classification Data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- **Size**: 45,000 records with 14 variables
- **Target Variable**: `loan_status` (1 = approved, 0 = rejected)

### Dataset Features

| Column | Description | Type |
|--------|-------------|------|
| person_age | Age of the person | Float |
| person_gender | Gender (female, male) | Categorical |
| person_education | Education level (Master, High School, Bachelor, Associate, Doctorate) | Categorical |
| person_income | Annual income | Float |
| person_emp_exp | Years of employment experience | Integer |
| person_home_ownership | Home ownership status (RENT, OWN, MORTGAGE, OTHER) | Categorical |
| loan_amnt | Loan amount requested | Float |
| loan_intent | Purpose of the loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION) | Categorical |
| loan_int_rate | Loan interest rate | Float |
| loan_percent_income | Loan amount as percentage of annual income | Float |
| cb_person_cred_hist_length | Length of credit history in years | Float |
| credit_score | Credit score of the person | Integer |
| previous_loan_defaults_on_file | Previous loan defaults indicator (No, Yes) | Categorical |
| loan_status | Loan approval status (1 = approved, 0 = rejected) | Integer |

## Methodology

### Model Selection: Binomial Regression (Logistic Regression)

**Reasons for choosing Binomial Regression:**
1. **Binary Outcome**: Designed for binary (yes/no) classification tasks
2. **Interpretability**: Provides interpretable coefficients explaining feature impact on approval probability
3. **Efficiency**: Computationally efficient and works well with large datasets
4. **Baseline Model**: Serves as a strong baseline for comparison with other models

### Data Preprocessing

1. **Exploratory Data Analysis (EDA)**: Comprehensive analysis of data distribution and relationships
2. **Feature Engineering**: One-hot encoding for categorical variables
3. **Data Splitting**: 80/20 train-test split
4. **Preprocessing Pipeline**: 
   - Numerical features: Mean imputation + Standard scaling
   - Categorical features: Mode imputation + One-hot encoding
5. **Dimensionality Reduction**: PCA applied to retain 90% of variance (reduced from 27 to 13 features)

### Model Performance

- **Accuracy**: 89.29%
- **ROC-AUC Score**: 95.16%
- **Precision (Class 0)**: 92%
- **Recall (Class 0)**: 94%
- **Precision (Class 1)**: 77%
- **Recall (Class 1)**: 74%

## Key Findings

1. **Class Imbalance**: The dataset shows class imbalance with more rejected loans than approved ones
2. **Feature Importance**: Previous loan defaults, credit score, and loan amount are significant predictors
3. **Model Performance**: The logistic regression model achieves high accuracy and excellent ROC-AUC score
4. **Dimensionality**: PCA successfully reduced features from 27 to 13 while retaining 91.5% of variance

## Files Structure

```
├── README.md                           # Project documentation
├── Stats_Model_FinalProject.ipynb     # Main analysis notebook
└── .gitignore                         # Git ignore file
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## Usage

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Open `Stats_Model_FinalProject.ipynb` in Jupyter Notebook
4. Run all cells to reproduce the analysis

## Results Summary

The binomial regression model successfully predicts loan approval with high accuracy (89.29%) and excellent discriminative ability (ROC-AUC: 95.16%). The model provides interpretable insights into the key factors influencing loan approval decisions, making it valuable for financial institutions to make informed and fair lending decisions.
