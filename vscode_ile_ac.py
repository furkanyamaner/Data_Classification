import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef

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
            dataFrame[column] = dataFrame[column].fillna(dataFrame[column].mean()) 
        else:
            # Fill missing categorical values with mode (most frequent value)
            dataFrame[column] = dataFrame[column].fillna(dataFrame[column].mode()[0])  
    return dataFrame

# Function to convert categorical columns to numeric
def changeToNumeric(dataFrame):
    for col in dataFrame.columns:
        if dataFrame[col].dtype not in ['float64', 'int64']:
            # Convert categorical columns to numeric codes
            dataFrame[col] = dataFrame[col].astype('category').cat.codes
    return dataFrame

# Function to perform evaluation metrics calculation
def evaluateModel(model, X, y):
    # Stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Metrics lists
    auc_scores = []
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    mcc_scores = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Use .iloc[] for row indexing
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # Use .iloc[] for row indexing
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and Predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Handle Perceptron which doesn't have predict_proba
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)  # for AUC
        elif hasattr(model, 'decision_function'):
            # Use decision_function for Perceptron as a proxy for probabilities
            y_prob = model.decision_function(X_test_scaled)
            # Convert to probabilities using a sigmoid function (for binary classification)
            if len(np.unique(y)) == 2:
                y_prob = 1 / (1 + np.exp(-y_prob))  # Sigmoid for binary classification
            else:
                # For multi-class classification, we can use the softmax function
                # but let's skip AUC calculation for Perceptron in multi-class case
                y_prob = None  # Set it to None, we will not calculate AUC for multi-class models with decision_function
        
        # Calculate metrics
        if y_prob is not None:  # Only calculate AUC if y_prob is available
            try:
                if len(np.unique(y)) > 2:  # Multi-class classification
                    auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')  # Multi-class AUC
                else:  # Binary classification
                    auc_score = roc_auc_score(y_test, y_prob)
                auc_scores.append(auc_score)
            except ValueError:
                # If AUC calculation fails, append NaN (e.g., in case of empty predictions or other issues)
                auc_scores.append(np.nan)
        else:
            auc_scores.append(np.nan)  # If no probability estimate, append NaN
        
        # Calculate other metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        mcc_scores.append(matthews_corrcoef(y_test, y_pred))
    
    # Store the results for this model
    results = {
        "Model": model.__class__.__name__,
        "AUC": np.nan if np.isnan(np.mean(auc_scores)) else np.mean(auc_scores),  # Handle NaN AUC
        "Accuracy (CA)": np.mean(accuracy_scores),
        "F1-score": np.mean(f1_scores),
        "Precision": np.mean(precision_scores),
        "Recall": np.mean(recall_scores),
        "MCC": np.mean(mcc_scores)
    }
    
    return results

# List of models to evaluate
models = [
    LogisticRegression(random_state=42, max_iter=1000),
    SVC(random_state=42, probability=True),  # SVC with probability=True for AUC
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    AdaBoostClassifier(random_state=42, algorithm='SAMME'),
    GaussianNB(),
    KNeighborsClassifier(),
    Perceptron(random_state=42),
]

# Clean the data
cleanedData = cleanData(dFrame)

# Convert non-numeric data to numeric
dFrame = changeToNumeric(cleanedData)

# Split data into features and target
X = dFrame.drop('GrainYield', axis=1)  # Assuming 'GrainYield' is the target column
y = dFrame['GrainYield']

# Initialize a list to store results
all_results = []

# Evaluate each model and store the results
for model in models:
    result = evaluateModel(model, X, y)
    all_results.append(result)

# Create a DataFrame to display results as a table
results_df = pd.DataFrame(all_results)

# Display the results
print("Model Evaluation Results:")
print(results_df)

# Sort by any column, by AUC
results_df = results_df.sort_values(by="AUC", ascending=False)
print("\nSorted by AUC:")
print(results_df)
