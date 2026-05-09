import pandas as pd
from sklearn.model_selection import train_test_split

from knowledge_base import rule_based_decision
from bayesian_model import train_bayes, predict_bayes
from ml_model import train_ml_model, predict_ml
from search_algorithm import rank_applicants
from evaluation import evaluate, plot_confusion, compare_models

# =========================
# EXTRA AI LOGIC FUNCTIONS
# =========================

def explain_prediction(applicant):
    reasons = []

    if applicant['Credit_History'] == 1:
        reasons.append("Good credit history")

    if applicant['ApplicantIncome'] > 40000:
        reasons.append("High income")

    if applicant['LoanAmount'] < 200:
        reasons.append("Low loan amount")

    return reasons


def risk_score(applicant):
    score = 0

    if applicant['Credit_History'] == 1:
        score += 50

    score += applicant['ApplicantIncome'] / 1000
    score -= applicant['LoanAmount'] / 10

    return score


def final_decision(rule, ml_pred, bayes_pred):
    if ml_pred == 1 and bayes_pred == 1:
        return "Approved (High Confidence)"
    elif ml_pred == 0 and bayes_pred == 0:
        return "Rejected (High Confidence)"
    else:
        return f"Uncertain → Rule Suggests: {rule}"


# =========================
# LOAD DATA
# =========================
data = pd.read_csv("dataset.csv")

# =========================
# PREPROCESSING
# =========================
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

data.fillna({
    'Gender': data['Gender'].mode()[0],
    'Married': data['Married'].mode()[0],
    'Dependents': data['Dependents'].mode()[0],
    'Self_Employed': data['Self_Employed'].mode()[0],
    'LoanAmount': data['LoanAmount'].mean(),
    'Loan_Amount_Term': data['Loan_Amount_Term'].mode()[0],
    'Credit_History': data['Credit_History'].mode()[0]
}, inplace=True)

original_data = data.copy()

data = pd.get_dummies(data, drop_first=True)

X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN
# =========================
bayes_model = train_bayes(X_train, y_train)
ml_model, scaler = train_ml_model(X_train, y_train)

# =========================
# PREDICT
# =========================
bayes_pred = predict_bayes(bayes_model, X_test)
ml_pred = predict_ml(ml_model, scaler, X_test)

# =========================
# EVALUATION
# =========================
metrics = {}
metrics["Bayesian"] = evaluate(y_test, bayes_pred, "Bayesian Model")
metrics["Logistic"] = evaluate(y_test, ml_pred, "Logistic Regression")

plot_confusion(y_test, bayes_pred, "Bayesian Confusion Matrix")
plot_confusion(y_test, ml_pred, "Logistic Confusion Matrix")

compare_models(metrics)

# =========================
# AI DECISION DEMO
# =========================
sample = original_data.iloc[0].to_dict()

rule = rule_based_decision(sample)
ml_sample = ml_pred[0]
bayes_sample = bayes_pred[0]

decision = final_decision(rule, ml_sample, bayes_sample)
reasons = explain_prediction(sample)
risk = risk_score(sample)

print("\n===== FINAL AI DECISION =====")
print("Decision:", decision)
print("Reasons:", reasons)
print("Risk Score:", risk)

# =========================
# SEARCH
# =========================
applicants = original_data.to_dict(orient='records')
ranked = rank_applicants(applicants)

print("\nTop Ranked Applicant:")
print(ranked[0])