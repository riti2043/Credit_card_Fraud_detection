# Credit Card Fraud Detection – End-to-End Machine Learning Web Application

## Overview
Credit card fraud is a major challenge in digital payment systems and financial transactions. As online payments continue to grow, detecting fraudulent activity quickly and accurately has become essential.

This project demonstrates a complete end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The system analyzes transaction attributes such as amount, merchant information, location, and transaction time to determine whether a transaction is legitimate or fraudulent.

The trained machine learning model is integrated into a Flask web application that allows users to input transaction details and receive real-time fraud predictions.

---

## Tech Stack

Python  
NumPy  
Pandas  
scikit-learn  
XGBoost  
imbalanced-learn (SMOTE)  

Visualization:  
Matplotlib  
Seaborn  

Deployment:  
Flask  
Gunicorn  

Development Tools:  
VS Code  
GitHub  

---

## Project Workflow

Data Exploration  
Data Cleaning and Validation  
Feature Engineering  
Handling Class Imbalance using SMOTE  
Model Training  
Model Evaluation  
Model Serialization  
Flask Web Application Deployment  

---

## Dataset Description

The dataset contains over sixty-five thousand simulated credit card transactions designed to represent real-world transaction patterns.

Each transaction includes the following information:

TransactionID  
TransactionDate  
Amount  
MerchantID  
TransactionType  
Location  
IsFraud  

The target variable `IsFraud` indicates whether the transaction is fraudulent or legitimate.

Fraudulent transactions represent approximately one percent of the dataset, making this a highly imbalanced classification problem.

---

## Data Preprocessing

Several preprocessing steps were performed before model training.

The transaction date was converted into datetime format and additional time-based features were extracted such as transaction hour, transaction day, and transaction month.

Categorical variables such as transaction type and location were encoded using one-hot encoding.

Invalid or incomplete records were removed and numerical features were scaled using StandardScaler.

---

## Handling Class Imbalance

Fraud detection datasets typically contain far fewer fraudulent transactions compared to legitimate ones.

To address this issue, SMOTE (Synthetic Minority Oversampling Technique) was applied to the training dataset. This method generates synthetic samples of the minority class to help the model learn patterns associated with fraudulent transactions.

---

## Machine Learning Models

Multiple machine learning models were implemented and compared during the experimentation phase.

Logistic Regression  
Random Forest  
XGBoost  
Isolation Forest  

These models were evaluated to determine which approach performs best for detecting fraudulent transactions.

---

## Evaluation Metrics

Since fraud detection datasets are highly imbalanced, accuracy alone is not a reliable metric.

The models were evaluated using the following metrics:

Precision  
Recall  
F1 Score  
Confusion Matrix  

These metrics provide better insight into how well the model identifies fraudulent transactions.

---

## Model Deployment

The trained model and preprocessing components were serialized using pickle.

The following files are used by the Flask application:

fraud_model.pkl  
scaler.pkl  
features.json  

A Flask application was created to serve predictions through a web interface and API endpoint.

Users can input transaction details through the web form and receive real-time predictions indicating whether a transaction is fraudulent or legitimate.

---
