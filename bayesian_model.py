from sklearn.naive_bayes import GaussianNB

def train_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def predict_bayes(model, X_test):
    return model.predict(X_test)