# Credit Card Default Prediction - README

Completed February 2022 as an extension of a project from my Master's of Data Science, demonstrating an end to end supervized learning workflow for classification.

## Overview

This project aims to develop a machine learning model to predict the likelihood of customers defaulting on their credit card payments in the next month. It utilizes the "Credit Card Clients Dataset" from Kaggle, which includes data on credit card usage and demographic variables.

## Problem Statement

The primary goal is to identify customers likely to default on their payments, a classic binary classification problem. The target variable, initially labeled as `default.payment.next.month`, has been simplified to `default` for this analysis, with `1` indicating a default and `0` denoting no default.

## Data Overview

The dataset exhibits a significant class imbalance, with a higher proportion of non-default cases. This poses a challenge as it may affect the model's ability to accurately predict the minority class (defaults). The project treats this as an anomaly detection problem, where precision (minimizing false positives) and recall (minimizing false negatives) are more critical than mere accuracy.

## Evaluation Metric

The model evaluation focuses on the F1-score to balance precision and recall. However, given the business context of minimizing financial loss due to defaults, more emphasis is placed on recall to capture as many default cases as possible without overly penalizing the precision.

## Methodology

### Data Preparation

Data preprocessing steps include handling missing values, encoding categorical variables, and normalizing numerical features. The dataset is split into training and testing sets to evaluate the model's performance.

### Feature Engineering

New features are created to better capture the financial behavior of customers, such as aggregating total bill amounts and payment amounts, and calculating the maximum payment delay.

### Model Selection

Several models are tested including logistic regression, naive Bayes, random forest, and LightGBM. Each model is evaluated using cross-validation on the training data.

### Hyperparameter Tuning

Models are fine-tuned using techniques like grid search and randomized search to find the optimal settings that maximize the F1-score.

### Model Evaluation

The best-performing model is selected based on cross-validation results and then finally evaluated on the unseen test set to assess its generalization capability.

## Key Libraries and Tools

- Pandas and NumPy for data manipulation
- Matplotlib and Altair for data visualization
- Scikit-learn for modeling and evaluation
- ELI5 and SHAP for model interpretation

## Conclusion

The project successfully develops a predictive model that can identify potential credit card defaults with reasonable accuracy and recall. The final model can be utilized by financial institutions to preemptively address or mitigate the risk of defaults, potentially leading to more stable revenue streams and reduced financial losses.

For more details, please refer to the Jupyter notebook provided in this repository.
