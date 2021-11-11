import joblib
from lightgbm import LGBMClassifier

from src.classifier.interface import IClassifier


class LGBMClf(IClassifier):
    def __init__(self, n_estimators, learning_rate, max_depth, early_stopping_rounds):
        self.model = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth
        )
        self.fitted = False
        self.early_stopping_rounds=early_stopping_rounds

    def train(self, X, y, X_test, y_test):
        self.model.fit(
            X=X,
            y=y,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.fitted = True

    def predict_proba(self, batch):
        assert self.fitted
        return self.model.predict_proba(batch)

    def predict(self, batch):
        assert self.fitted
        return self.model.predict(batch)

    def save(self, filename):
        print(f"=== Saving LGBM (Shallow ML) : {filename} ===")
        joblib.dump(self.model, filename)

    def load(self, filename):
        print("=== Loading LGBM (Shallow ML) : {filename} === ")
        self.model = joblib.load(filename)
        self.fitted = True
