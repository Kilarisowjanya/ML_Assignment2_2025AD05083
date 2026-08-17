"""
STREAMLIT APPLICATION
ML Assignment-2 | Sowjanya Kilari | BITS ID: 2025AD05083

Run:
    python3 -m streamlit run app.py

The app:
    1. Uploads test CSV data.
    2. Applies the saved preprocessing pipeline.
    3. Lets the user choose one of five models.
    4. Displays predictions.
    5. Displays Accuracy, AUC, Precision, Recall, F1 and MCC when Churn
       is included.
    6. Displays a confusion matrix and classification report.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score
)


# ---------------------------------------------------------------------------
# 1. APPLICATION CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Telco Customer Churn",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------------------------
# 2. LOAD SAVED MODELS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load the five trained models and the fitted preprocessor."""
    filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
    }

    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in filenames.items()
    }

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    return models, preprocessor


models, preprocessor = load_artifacts()


# ---------------------------------------------------------------------------
# 3. USER INTERFACE HEADER
# ---------------------------------------------------------------------------
st.title("📊 Telco Customer Churn Prediction")
st.write(
    "**ML Assignment-2 | Sowjanya Kilari | BITS ID: 2025AD05083**"
)

st.markdown(
    """
    This application compares five machine-learning classification models
    for predicting customer churn.
    """
)


# ---------------------------------------------------------------------------
# 4. MODEL SELECTION
# ---------------------------------------------------------------------------
selected_model_name = st.sidebar.selectbox(
    "Select a model",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.sidebar.write(
    f"**Selected model:** {selected_model_name}"
)


# ---------------------------------------------------------------------------
# 5. CSV UPLOAD
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload the assignment test CSV to continue.")
    st.stop()

data = pd.read_csv(uploaded_file)

st.subheader("Uploaded Data")
st.write(
    f"Shape: **{data.shape}**"
)
st.dataframe(
    data.head(10),
    width="stretch"
)


# ---------------------------------------------------------------------------
# 6. VALIDATE REQUIRED FEATURES
# ---------------------------------------------------------------------------
expected_features = list(
    preprocessor.feature_names_in_
)

missing = [
    feature
    for feature in expected_features
    if feature not in data.columns
]

if missing:
    st.error("Required columns are missing.")
    st.write(missing)
    st.stop()


# ---------------------------------------------------------------------------
# 7. SEPARATE INPUT FEATURES AND OPTIONAL TARGET
# ---------------------------------------------------------------------------
has_target = "Churn" in data.columns

if has_target:
    X_input = data.drop(
        columns=["Churn"]
    ).copy()

    y_true = data["Churn"].map(
        {"No": 0, "Yes": 1}
    )

    # Also accept numeric 0/1 target values.
    if y_true.isna().any():
        y_true = pd.to_numeric(
            data["Churn"],
            errors="coerce"
        )

    if y_true.isna().any():
        st.error(
            "Churn must contain No/Yes or 0/1 values."
        )
        st.stop()

    y_true = y_true.astype(int)

else:
    X_input = data.copy()
    y_true = None


# ---------------------------------------------------------------------------
# 8. APPLY THE SAVED PREPROCESSOR
# ---------------------------------------------------------------------------
# Only transform is used here because the preprocessor was already fitted
# during training. We must NOT fit it again on user/test data.
X_processed = preprocessor.transform(
    X_input[expected_features]
)


# ---------------------------------------------------------------------------
# 9. GENERATE PREDICTIONS
# ---------------------------------------------------------------------------
y_pred = selected_model.predict(
    X_processed
)

prediction_table = X_input.copy()
prediction_table["Predicted_Churn"] = np.where(
    y_pred == 1, "Yes", "No"
)

st.subheader(
    f"Predictions — {selected_model_name}"
)
st.dataframe(
    prediction_table.head(20),
    width="stretch"
)


# ---------------------------------------------------------------------------
# 10. EVALUATE THE SELECTED MODEL
# ---------------------------------------------------------------------------
if has_target:
    if hasattr(
        selected_model,
        "predict_proba"
    ):
        y_probability = selected_model.predict_proba(
            X_processed
        )[:, 1]
    else:
        y_probability = selected_model.decision_function(
            X_processed
        )

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probability),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(
            y_true, y_pred
        )
    }

    st.subheader(
        f"Evaluation Metrics — {selected_model_name}"
    )

    columns = st.columns(6)

    for column, (name, value) in zip(
        columns, metrics.items()
    ):
        column.metric(
            name,
            f"{value:.4f}"
        )


    # -----------------------------------------------------------------------
    # 11. CONFUSION MATRIX
    # -----------------------------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    image = ax.imshow(cm)

    ax.set_title(
        f"Confusion Matrix — {selected_model_name}"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                cm[row, col],
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    st.pyplot(
        fig,
        clear_figure=True
    )


    # -----------------------------------------------------------------------
    # 12. CLASSIFICATION REPORT
    # -----------------------------------------------------------------------
    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        width="stretch"
    )


    # -----------------------------------------------------------------------
    # 13. COMPARE ALL FIVE MODELS
    # -----------------------------------------------------------------------
    st.subheader(
        "All-Model Performance Comparison"
    )

    comparison = []

    for name, model in models.items():
        predictions = model.predict(
            X_processed
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                X_processed
            )[:, 1]
        else:
            probabilities = model.decision_function(
                X_processed
            )

        comparison.append({
            "ML Model": name,
            "Accuracy": accuracy_score(
                y_true, predictions
            ),
            "AUC": roc_auc_score(
                y_true, probabilities
            ),
            "Precision": precision_score(
                y_true, predictions,
                zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions,
                zero_division=0
            ),
            "F1": f1_score(
                y_true, predictions,
                zero_division=0
            ),
            "MCC": matthews_corrcoef(
                y_true, predictions
            )
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    st.dataframe(
        comparison_df.round(4),
        width="stretch"
    )

else:
    st.warning(
        "Churn is not present, so evaluation metrics cannot be calculated."
    )


# ---------------------------------------------------------------------------
# 14. DOWNLOAD PREDICTIONS
# ---------------------------------------------------------------------------
csv_output = prediction_table.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Predictions CSV",
    data=csv_output,
    file_name="predictions.csv",
    mime="text/csv"
)
