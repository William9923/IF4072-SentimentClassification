import numpy as np

from tensorflow.keras.layers import (
    Input,
    Dropout,
    Dense,
    Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from transformers import TFRobertaModel

from src.classifier.interface import IClassifier


class FineTuneRobertaClf(IClassifier):
    def __init__(
        self,
        batch_size,
        length,
        epochs,
        model_name,
        loss,
        optimizer,
        metrics,
    ):

        self.batch_size = batch_size
        self.bert = TFRobertaModel.from_pretrained(model_name)

        self.input_ids = Input(shape=(length,), name="input_ids", dtype="int32")
        self.attention_mask = Input(
            shape=(length), name="attention_mask", dtype="int32"
        )

        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }

        self.embedding = self.bert.roberta(inputs)[0]
        self.flatten = Flatten()(self.embedding)
        self.drop = Dropout(0.3)(self.flatten)
        self.out = Dense(3, activation="softmax")(self.drop)

        self.model = Model(inputs=inputs, outputs=self.out)

        self.fitted = False
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.epochs = epochs

    def train(self, X, y, X_test, y_test):
        self.history = self.model.fit(
            x=X,
            y=y,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
        )
        self.fitted = True

    def summary(self):
        return self.model.summary()

    def predict_proba(self, batch):
        assert self.fitted
        return self.model.predict(x=batch, batch_size=self.batch_size)

    def predict(self, batch):
        return np.argmax(self.predict_proba(batch), axis=-1)

    def save(self, filename):
        print(f"=== Saving Fine Tuned RoBerta Model : {filename} ===")
        self.model.save(filename)

    def load(self, filename):
        print(f"=== Loading Fine Tuned RoBerta Model : {filename} === ")
        self.model = load_model(filename)
        self.fitted = True
