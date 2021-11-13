import joblib
from sklearn.naive_bayes import GaussianNB

from src.classifier.interface import IClassifier


class NaiveBayesClf(IClassifier):
    def __init__(self):
        self.model = GaussianNB()
        self.fitted = False

    def train(self, X, y, X_test, y_test):
        self.model.fit(
            X=X,
            y=y,
        )
        self.fitted = True

    def summary(self):
        summary = f"NB Configs :\n {str(self.model)}"
        return summary

    def predict_proba(self, batch):
        assert self.fitted
        return self.model.predict_proba(batch)

    def predict(self, batch):
        assert self.fitted
        return self.model.predict(batch)

    def save(self, filename):
        print(f"=== Saving NB (Shallow ML) : {filename} ===")
        joblib.dump(self.model, filename)

    def load(self, filename):
        print("=== Loading NB (Shallow ML) : {filename} === ")
        self.model = joblib.load(filename)
        self.fitted = True
