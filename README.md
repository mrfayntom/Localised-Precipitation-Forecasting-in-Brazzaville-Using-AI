# Brazzaville Precipitation Prediction Challenge 

## Challenge Overview

This hackathon aims to develop a **machine learning model** capable of predicting **daily or sub-daily precipitation levels** in Brazzaville by leveraging meteorological and spatio-temporal data. The goal is to provide accurate, actionable forecasts that can help mitigate the impact of unpredictable rainfall and improve decision-making for the city's residents and planners.

By addressing this complex climate problem, we hope to contribute tools that support sustainable urban development and resilience against climate-related disasters in the region.

---

## Dataset and Submission Files Overview 

I review these file for you so you have an overview

| **File Name**             | **Description**                                                                                                                                                  | **Size**    |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| `SampleSubmission.csv`    | An example submission file showing the exact format required. The order of rows doesn’t matter, but the "ID" column names must be correct.                      | 31.5 KB     |
| `Test_data.csv`           | Test dataset similar to the training data but **without the target variable**. Use this to generate predictions with your trained model.                        | 120.4 KB    |
| `Train_data.csv`          | Training dataset containing both features and the target (precipitation levels). This is what you’ll use to train your machine learning model.                   | 297.7 KB    |
| `VariableDefinitions.csv` | Descriptions and explanations of the variables in both the train and test datasets to help you understand the data features.                                    | 361 B       |
| `Starter_notebook.ipynb`  | A starter notebook with example code to help you load the data, perform initial analysis, and make your first submission on the Zindi platform.                  | 114.6 KB    |

---

## Overview of `analyze.py` 

This script performs an initial exploratory data analysis (EDA) on the training dataset to better understand the distribution and patterns of rainfall in Brazzaville.

- **Data Loading & Summary:**  
  Reads the training data and prints basic information including data types, missing values, and a sample preview of the dataset.

- **Target Variable Distribution:**  
  Visualizes the distribution of the rainfall target variable using histograms — both in its original scale and log-transformed scale — to highlight skewness and variability.

- **Feature Correlations:**  
  Computes and displays a heatmap of correlations between features (excluding ID and DATE) and the target variable, helping identify which features might be most predictive.

- **Temporal Analysis:**  
  Extracts the month from the date column and plots a boxplot to reveal seasonal trends in monthly rainfall patterns.

---
## Overview of `pipeline.py` 

This script builds a comprehensive machine learning pipeline for predicting precipitation levels using the training data.

- **Feature Engineering:**  
  Creates new meaningful features derived from the original meteorological data, including seasonal indicators, interaction terms, and climate clusters to better capture complex weather patterns.

- **Data Preparation:**  
  Splits the dataset into training and validation sets, applying log transformation to the target variable to handle skewness.

- **Model Training:**  
  Trains multiple regression models (including XGBoost, LightGBM, CatBoost, and Random Forest) to predict rainfall levels, each with carefully chosen hyperparameters.

- **Ensembling:**  
  Combines the predictions from the individual models using a Bayesian Ridge regression to produce a more robust and accurate final forecast.

- **Evaluation:**  
  Calculates and reports the Root Mean Squared Error (RMSE) for each model and the ensemble on the validation set to measure performance.

---
## Brazzaville Precipitation Prediction Challenge 

**Hackathon Link:**  
[Localised Precipitation Forecasting in Brazzaville](https://zindi.africa/competitions/localised-precipitation-forecasting-in-brazzaville-in-republic-of-congo-using-ai/data)

Here is the video of my rank and about hackathon

[![Watch my hackathon rank video](https://img.youtube.com/vi/UrqkW1ZheS8/maxresdefault.jpg)](https://youtu.be/UrqkW1ZheS8)

---
### Personal Note

I wanted to share that I had to withdraw from this hackathon about 10 days ago due to some personal challenges. I'm currently dealing with urgent issues related to SAT and fee waiver deadlines, which has been quite stressful. Unfortunately, I haven’t been able to secure the fee waiver yet, and with the deadline approaching fast, it has been hard for me to focus on the hackathon.

Additionally, this challenge is specifically open only to Congolese nationals or foreign students at Congolese universities, requiring proof of nationality for winners. Given these circumstances, I decided to step back from the competition.

I hope to return to similar challenges in the future once things settle down. Thanks for understanding!

---
