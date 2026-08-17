"""Telco Customer Churn Classification
ML Assignment-2
Name: Sowjanya Kilari
BITS ID: 2025AD05083

This script contains the machine-learning workflow used in the assignment:
data loading, cleaning, feature preparation, preprocessing, model training,
evaluation, comparison, and saving the trained artifacts.

The Streamlit deployment interface is implemented separately in app.py.
"""

# Import Required Libraries
# Purpose: import required libraries for the Telco churn classification workflow.

import pandas as pd
import numpy as np

#Telco Customer Churn Dataset which consists of 7043 rows data and 21 Features. 
# Load the Telco Customer Churn Dataset

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Rows and columns:", df.shape)
print("\n Total Column names:")
print(df.columns.tolist())
print("\n", "*"*30) 
print("\nfirst 5 records of data")

# Inspect Dataset Structure and Basic Statistics
print(df.head())
print("\n info of a file ", "*"*30) 
print(df.info())
print("*"*30)
print(df.describe())

# Check and Clean Missing Values
print(df["Churn"].value_counts())
print("\n churn percentage ",df["Churn"].value_counts(normalize=True) * 100)

print(df["TotalCharges"].value_counts().head(10))

print("Blank TotalCharges:",
      (df["TotalCharges"].str.strip() == "").sum())

df_clean = df.copy()
df_clean["TotalCharges"] = pd.to_numeric(
    df_clean["TotalCharges"],
    errors="coerce"
)

# Encode the Churn Target

print(df_clean["TotalCharges"].dtype)

print("Missing TotalCharges:",
      df_clean["TotalCharges"].isna().sum())

# Prepare Features and Target Variable
df_clean = df_clean.dropna(subset=["TotalCharges"])


print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)
print("Missing values in TotalCharges:",
      df_clean["TotalCharges"].isna().sum())

missing = df_clean.isnull().sum()
print(missing[missing > 0])

print(missing)
cleaned_data = df_clean.copy()


categorical_columns = cleaned_data.select_dtypes(
    include="str"
).columns

print(categorical_columns.tolist())

X= cleaned_data.drop(columns=["Churn","customerID"])
y=cleaned_data["Churn"]

print("X shape:", X.shape)
print("y shape:", y.shape)


categorical_features = X.select_dtypes(
    include="str"
).columns.tolist()

print("Number of categorical features:", len(categorical_features))
print("\nCategorical features:")
for column in categorical_features:
    print("-", column)


numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\n Number of numerical features:", len(numerical_features))
print("\n Numerical features")

for column in numerical_features:
    print("-", column)


print(y.value_counts())

y_encoded= y.map({
    "No": 0,
    "Yes": 1,
})

print("Encoded target Distribution")
print(y_encoded.value_counts())

print("\n First 10 tragte values")
print(y_encoded.head(10))



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)



X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed X_train shape:", X_train_processed.shape)
print("Processed X_test shape:", X_test_processed.shape)



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

print("All required models imported successfully.")


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

print("Number of models:", len(models))
print("Models:", list(models.keys()))


for name, model in models.items():
    model.fit(X_train_processed, y_train)
    print(f"{name} trained successfully.")



from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import pandas as pd

results = []

for name, model in models.items():

    # Predictions
    y_pred = model.predict(X_test_processed)

    # Probability/score for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_processed)[:, 1]
    else:
        y_prob = model.decision_function(X_test_processed)

    # Calculate required metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "ML Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

results_df = pd.DataFrame(results)

print("Model Performance Comparison:")
print(results_df.round(4).to_string(index=False))



# Identify the best model for each metric

metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]

for metric in metrics:
    best_index = results_df[metric].idxmax()
    best_model = results_df.loc[best_index, "ML Model"]
    best_score = results_df.loc[best_index, metric]

    print(f"{metric}: {best_model} ({best_score:.4f})")


import matplotlib.pyplot as plt

results_plot = results_df.set_index("ML Model")

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("ML Model Performance Comparison")
plt.xlabel("ML Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

for name, model in models.items():

    y_pred = model.predict(X_test_processed)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()



import os
import joblib

# Create folder to store trained models
os.makedirs("models", exist_ok=True)

# Save all trained models
for name, model in models.items():
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"models/{filename}")

# Save the preprocessing pipeline
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("All models and preprocessor saved successfully.")



print("Saved files:")

for file in os.listdir("models"):
    print("-", file)



# Create test dataset for Streamlit

test_data = X_test.copy()

# Add the target column
test_data["Churn"] = y_test.values

# Save as CSV
test_data.to_csv("test_data.csv", index=False)

print("test_data.csv created successfully.")
print("Shape:", test_data.shape)
print("\nColumns:")
print(test_data.columns.tolist())


# Verify the saved test CSV

test_check = pd.read_csv("test_data.csv")

print("Test CSV shape:", test_check.shape)
print("\nFirst 5 rows:")
print(test_check.head().to_string(index=False))


print("Features expected by preprocessor:")
print(preprocessor.feature_names_in_)

print("\nNumber of input features:",
      len(preprocessor.feature_names_in_))

