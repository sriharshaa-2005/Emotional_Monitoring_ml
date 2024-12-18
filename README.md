# Emotional_Monitoring_ml


# Emotional Monitoring System  

This repository contains a machine learning project aimed at monitoring and predicting emotional states and engagement levels based on biometric and behavioral data. The system leverages multiple machine learning models to classify emotional states and engagement levels, helping individuals or organizations gain insights into emotional and cognitive health.

---

## Table of Contents  

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Dataset Description](#dataset-description)  
- [Machine Learning Models](#machine-learning-models)  
- [Evaluation](#evaluation)  
- [Results](#results)  
- [How to Run the Project](#how-to-run-the-project)  
- [Future Work](#future-work)  
- [License](#license)  

---

## Project Overview  

This project analyzes biometric data such as heart rate, EEG activity, and pupil diameter to predict emotional states (e.g., engaged, partially engaged, or disengaged) and engagement levels. Machine learning models like Decision Trees, Random Forest, and Gradient Boosting Classifiers are used for classification tasks.  

---

## Features  

- **Data Preprocessing:** Handles missing values, outliers, and encodes categorical variables.  
- **Exploratory Data Analysis (EDA):** Visualizations to understand relationships between features and target variables.  
- **Model Training:** Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting Classifier are implemented.  
- **Model Evaluation:** Models are evaluated using accuracy and visualized for performance comparison.  
- **Prediction:** Supports real-time predictions using trained models.  

---

## Technologies Used  

- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)  
- Jupyter Notebook or Google Colab for development  
- Machine Learning Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, K-Nearest Neighbors  

---

## Dataset Description  

The dataset contains 1000 rows with the following columns:  

- **HeartRate:** Measured in beats per minute.  
- **SkinConductance:** Electrical conductance of the skin.  
- **EEG:** Electrical activity in the brain.  
- **Temperature:** Body temperature.  
- **PupilDiameter:** Diameter of the pupil.  
- **SmileIntensity/FrownIntensity:** Intensity of facial expressions.  
- **CortisolLevel:** Hormonal levels linked to stress.  
- **ActivityLevel, AmbientNoiseLevel, LightingLevel:** Environmental factors.  
- **EmotionalState:** Categories - `engaged`, `partially engaged`, `disengaged`.  
- **CognitiveState:** Categories - `focused`, `distracted`.  
- **EngagementLevel:** Target variable representing engagement intensity.  

---

## Machine Learning Models  

The following models were implemented and evaluated:  

- **Logistic Regression:** Accuracy = 70%  
- **Support Vector Machine (SVM):** Accuracy = 54.5%  
- **K-Nearest Neighbors (KNN):** Accuracy = 60.5%  
- **Decision Tree Classifier:** Accuracy = 99.5%  
- **Random Forest Classifier:** Accuracy = 99.5%  
- **Gradient Boosting Classifier:** Accuracy = 99.5%  

---

## Evaluation  

The models were evaluated using accuracy scores. The Decision Tree, Random Forest, and Gradient Boosting Classifiers performed the best with an accuracy score of 99.5%.  

---

## Results  

- **Best Models:** Decision Tree, Random Forest, Gradient Boosting  
- **Key Insights:** Environmental and biometric data can strongly predict emotional and engagement states.  

---


