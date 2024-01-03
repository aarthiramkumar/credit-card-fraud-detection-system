# credit-card-fraud-detection-system

# Credit Card Fraud Detection Model

## Overview

This project aims to develop a robust Credit Card Fraud Detection Model using machine learning techniques. The dataset used for training and evaluation is sourced from Kaggle, and the model is built through various stages including Exploratory Data Analysis (EDA), feature selection, data preprocessing, and model training.

## Problem Statement

Credit card fraud is a significant concern in the financial industry. Detecting fraudulent transactions is crucial to prevent financial losses and protect consumers. Traditional rule-based systems often fall short in identifying sophisticated fraud patterns. This project addresses the challenge of accurately detecting fraudulent credit card transactions using machine learning.

## Dataset

The dataset used for this project is obtained from Kaggle and can be found [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains transactions made by credit cards in September 2013 and includes both legitimate and fraudulent transactions.

## Project Structure

- **Notebooks**: Contains Jupyter notebooks used for data analysis, feature selection, and model development.
  - `1_EDA.ipynb`: Exploratory Data Analysis on the credit card transactions dataset.
  - `2_Feature_Selection.ipynb`: Feature selection process to identify relevant features.
  - `3_Model_Training.ipynb`: Building and training the Credit Card Fraud Detection Model.

- **Data**: Contains the dataset used for training and evaluation (`creditcard.csv`).

- **Models**: Contains the trained model file (`frauddetection.pkl`).

- **App**: Contains the Python application for credit card fraud detection.
  - `app.py`: FastAPI application to predict transaction legitimacy.

- **Requirements**: Lists the necessary dependencies for running the model and the application (`requirements.txt`).

## Technology Stack

- **Python**: Primary programming language for data analysis, model development, and application implementation.
- **Pandas and NumPy**: Data manipulation and numerical operations.
- **Scikit-Learn**: Machine learning library for model training and evaluation.
- **FastAPI**: Framework for building the API endpoint in the Credit Card Fraud Detection Application.
- **Streamlit**: Used for creating the user interface for interacting with the model.
- **Joblib**: Serialization library for saving and loading machine learning models.
- **Jupyter Notebooks**: Used for interactive data analysis and model development.

## Model Development

The model development process involves the following steps:

1. **Exploratory Data Analysis (EDA):** Understand the characteristics of the dataset, identify patterns, and gain insights into the features.

2. **Feature Selection:** Select relevant features for building a robust fraud detection model.

3. **Data Preprocessing:** Handle missing values, scale features, and preprocess the dataset for training.

4. **Model Training:** Train the Credit Card Fraud Detection Model using machine learning algorithms. In this project, Logistic Regression is used, but other algorithms can also be explored.

5. **Model Evaluation:** Assess the model's performance using metrics such as accuracy, precision, recall, and F1-score.

## Application

The Python application (`app.py`) uses the trained model to predict the legitimacy of credit card transactions. It provides a FastAPI endpoint (`/predict_api/`) to receive input features and return predictions.

### How to Run the Application

1. Install dependencies using `pip install -r requirements.txt`.
2. Run the FastAPI application using `uvicorn app:app --host 127.0.0.1 --port 7000`.
3. Access the application at `http://127.0.0.1:7000` in your web browser.

## Usage

To use the Credit Card Fraud Detection Model:

1. Provide input features (Year, Month, UseChip, Amount, MerchantName, MerchantCity, MerchantState, mcc, etc.).
2. Click the "Submit" button on the Streamlit web interface.
3. Receive predictions on whether the transaction is legitimate or fraudulent.

## Acknowledgments

- The dataset used in this project is sourced from Kaggle: [link-to-the-dataset](lhttps://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- This project is inspired by the need for effective credit card fraud detection in financial transactions.



