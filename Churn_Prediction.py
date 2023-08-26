import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Load the dataset for training
data = pd.read_csv('/Users/apple/Documents/Projects/Customer_Churn_Prediction/customer_churn_large_dataset.csv')

# Sample a smaller portion of the data for development
data = data.sample(frac=0.01, random_state=42)

# Data Preprocessing
numerical_columns = ['Age', 'Monthly_Bill', 'Total_Usage_GB']

# Feature Engineering
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

X = data.drop('CustomerID', 'Name', 'Location'], axis=1)  # Features
y = data['Churn']  # Target variable

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building with Support Vector Machine (SVM)
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

svm_classifier = SVC(random_state=42)
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_svm_model = grid_search.best_estimator_
best_svm_model.fit(X_train, y_train)

# Save the best SVM model for deployment
feature_names = list(X.columns)  # Get the feature names used during training
joblib.dump(feature_names, 'feature_names.pkl')
joblib.dump(best_svm_model, 'churn_model_svm_improved.pkl')


# print("Support Vector Machine (SVM) Model Metrics:")
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
# print("Confusion Matrix:")
# print(conf_matrix)
