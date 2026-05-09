from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_ml_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    return model, scaler

def predict_ml(model, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    return model.predict(X_test_scaled)