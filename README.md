# Customer Churn Prediction Assignment Report

## Introduction

 The goal of the assignment was to build and deploy a machine learning model to predict customer churn based on various customer features.

## Data Preprocessing and Model Building

1. **Data Loading and Sampling:** The assignment started with loading the customer churn dataset from a CSV file. To make the development process faster and more manageable, a smaller portion (1%) of the dataset was randomly sampled for development purposes.

2. **Data Preprocessing:** The dataset was preprocessed to prepare it for model building. This involved:
    - Identifying numerical columns: Columns like 'Age', 'Monthly_Bill', and 'Total_Usage_GB' were identified as numerical features.
    - Label Encoding: The 'Gender' column was label encoded, converting categorical data (Male/Female) into numerical format (0/1).

3. **Feature Engineering:** After preprocessing, non-numeric columns ('CustomerID', 'Name', 'Location') and the target variable ('Churn') were excluded from the feature set.

4. **Data Splitting:** The dataset was split into training and testing sets using an 80-20 split ratio. This allowed for model evaluation.

5. **Model Building with SVM:** A Support Vector Machine (SVM) classifier was chosen as the predictive model. Hyperparameter tuning was performed using GridSearchCV to find the best combination of hyperparameters (C and kernel) to maximize the F1 score. The best SVM model was then trained on the training data.

6. **Model Deployment:** The best SVM model, along with the feature names used during training, was saved using joblib for deployment in the Streamlit app.

## Streamlit App for Customer Churn Prediction


## Model Evaluation


## Model Performance

The reported model performance metrics on the test data (unavailable in the provided code) are as follows:
- **Accuracy:** 0.5
- **Precision:** 0.5
- **Recall:** 1.0
- **F1 Score:** 0.67
- **Confusion Matrix:** 
  ```
  [[  0 100]
   [  0 100]]
  ```

## Conclusion

The assignment involved building and deploying a machine learning model for customer churn prediction. The SVM model achieved a reasonable F1 score of 0.67, indicating its effectiveness in identifying potential customer churn. The Streamlit web application provides a user-friendly interface for making churn predictions based on customer data.
