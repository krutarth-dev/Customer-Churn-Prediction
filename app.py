import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained SVM model
model = joblib.load('/Users/apple/Documents/Projects/Customer_Churn_Prediction/churn_model_svm_improved.pkl')

# Load the feature names used during model training
feature_names = joblib.load('/Users/apple/Documents/Projects/Customer_Churn_Prediction/feature_names.pkl')

# Streamlit app title and description
st.title('Customer Churn Prediction App')
st.write('Enter customer information to predict churn.')

# Create input fields for user to enter customer data
st.sidebar.header('User Input')

# Create input fields for each feature using feature_names
input_data = {}
for feature in feature_names:
    if feature != 'Location':  # Exclude 'Location' feature
        if feature == 'Gender':
            gender_mapping = {'Male': 0, 'Female': 1}
            input_data[feature] = gender_mapping[st.sidebar.selectbox(feature, ['Male', 'Female'])]
        elif feature == 'Age':
            input_data[feature] = st.sidebar.slider(feature, 18, 100, 30)
        elif feature == 'Monthly_Bill' or feature == 'Total_Usage_GB':
            input_data[feature] = st.sidebar.number_input(f'{feature} Amount', min_value=0.0, step=1.0)

# Add 'Subscription_Length_Months' with a default value if it's missing
if 'Subscription_Length_Months' not in input_data:
    input_data['Subscription_Length_Months'] = 12  # You can change the default value as needed

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Ensure the order of features in input_df matches feature_names
input_df = input_df[feature_names]

# Define a function to make predictions
def predict_churn(data):
    prediction = model.predict(data)
    return prediction[0]

# Make predictions when the user clicks the "Predict" button
if st.sidebar.button('Predict'):
    churn_prediction = predict_churn(input_df)
    
    if churn_prediction == 1:
        st.sidebar.error('Prediction: Churn')
    else:
        st.sidebar.success('Prediction: Not Churn')
