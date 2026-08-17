# ML Assignment-2 — Telco Customer Churn

**Name:** Sowjanya Kilari  
**BITS ID:** 2025AD05083

## Project objective

This project predicts whether a telecommunications customer will churn. Five
classification models are trained and compared using the same stratified
train/test split.

### Models

1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest (Ensemble)

### Evaluation metrics

- Accuracy
- AUC
- Precision
- Recall
- F1
- MCC

## Dataset

Telco Customer Churn contains 7,043 records and 21 columns. `customerID` is
removed because it is an identifier. `TotalCharges` is converted to numeric;
11 rows with blank TotalCharges are removed, leaving 7,032 observations.

The target is encoded as:

- No -> 0
- Yes -> 1

The resulting data has 19 input features. An 80/20 stratified train/test split
with `random_state=42` is used, giving 5,625 training rows and 1,407 test rows.

Categorical variables are one-hot encoded and numerical variables are
standardized. The preprocessor is fitted only on training data and only
transformed on test/new data.

## Recorded assignment results

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8038 | 0.8359 | 0.6485 | 0.5722 | 0.6080 | 0.4795 |
| Decision Tree | 0.7356 | 0.6634 | 0.5026 | 0.5107 | 0.5066 | 0.3261 |
| KNN | 0.7605 | 0.7793 | 0.5468 | 0.5775 | 0.5618 | 0.3974 |
| Naive Bayes | 0.6823 | 0.8049 | 0.4472 | 0.8262 | 0.5803 | 0.4033 |
| Random Forest (Ensemble) | 0.7832 | 0.8133 | 0.6194 | 0.4786 | 0.5400 | 0.4069 |

### Analysis

- **Logistic Regression:** best overall balance; highest Accuracy, AUC,
  Precision, F1 and MCC.
- **Decision Tree:** weakest overall among the five on this test split.
- **KNN:** intermediate performance.
- **Naive Bayes:** highest Recall (0.8262), making it useful when identifying
  as many churners as possible is the priority, but it creates more false
  positives.
- **Random Forest:** intermediate overall performance with good Accuracy/AUC,
  but lower Recall.

**Overall winner:** Logistic Regression based on the majority of the required
metrics. Naive Bayes can be considered when maximum churn detection is more
important than precision.

## Streamlit application

Run:

```bash
python3 -m streamlit run app.py
```

The app supports CSV upload, model selection, prediction, evaluation metrics,
confusion matrix, classification report, all-model comparison and prediction
CSV download.

## Project structure

```text
ML_Assignment2_2025AD05083/
├── main.py
├── train_models.py
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_results.csv
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── 2025ad05083_ML_Assignment2.ipynb
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── preprocessor.pkl
```

## Git commit plan

Use multiple meaningful commits rather than one large commit:

```bash
git init
git add README.md requirements.txt .gitignore
git commit -m "docs: add project README and requirements"

git add main.py
git commit -m "feat: convert notebook workflow to Python"

git add train_models.py
git commit -m "feat: add reproducible model training pipeline"

git add app.py
git commit -m "feat: add Streamlit churn prediction app"

git add test_data.csv model_results.csv models/
git commit -m "chore: add test data and trained model artifacts"

git status
git log --oneline
```

Then connect the GitHub repository and push:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

**Do not commit secrets, API keys, passwords, or `.env` files.**
