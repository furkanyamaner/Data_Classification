import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

# Read data
dFrame = pd.read_excel("Data_processed.xlsx")

# Function to clean the data (fill missing values)
def cleanData(dataFrame):
    for column in dataFrame.columns:
        if dataFrame[column].dtype in ['float64', 'int64']:
            # Fill missing numeric values with mean
            dataFrame[column].fillna(dataFrame[column].mean(), inplace=True)
        else:
            # Fill missing categorical values with mode (most frequent value)
            dataFrame[column].fillna(dataFrame[column].mode()[0], inplace=True)
    return dataFrame

# Function to convert categorical columns to numeric
def changeToNumeric(dataFrame):
    for col in dataFrame.columns:
        if dataFrame[col].dtype not in ['float64', 'int64']:
            # Convert categorical columns to numeric codes
            dataFrame[col] = dataFrame[col].astype('category').cat.codes
    return dataFrame

# Function for model evaluation
def evaluateModel(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix: {model.__class__.__name__}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# List of models to evaluate
models = [
    LogisticRegression(random_state=42),
    SVC(random_state=42),
    DecisionTreeClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),  
]

# Clean the data
cleanedData = cleanData(dFrame)

# Convert non-numeric data to numeric
dFrame = changeToNumeric(cleanedData)

# Split data into features and target
X = dFrame.drop('GrainYield', axis=1)  # Assuming 'GrainYield' is the target column
y = dFrame['GrainYield']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluate each model in the list
for model in models:
    evaluateModel(model, X_train_scaled, X_test_scaled, y_train, y_test)
