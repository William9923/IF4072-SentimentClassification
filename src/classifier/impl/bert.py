import numpy as np

from tensorflow.keras.layers import (
    Input,
    Dropout,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from transformers import TFDistilBertModel

from src.classifier.interface import IClassifier

class FineTuneBertClf(IClassifier):
    def __init__(
        self,
        batch_size,
        length,
        model_name="distilbert-base-uncased",
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=3e-5),
        metrics=["accuracy"],
    ):

        self.batch_size = batch_size
        self.bert = TFDistilBertModel.from_pretrained(model_name)

        self.input_ids = Input(shape=(length,), name="input_ids", dtype="int32")
        self.attention_mask = Input(
            shape=(length), name="attention_mask", dtype="int32"
        )

        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }

        self.embedding = self.bert(inputs)[0]
        self.fcn1 = Dense(length, activation="relu")(self.embedding[:, 0, :])
        self.fcn2 = Dropout(0.5)(self.fcn1)
        self.out = Dense(1, activation="sigmoid")(self.fcn2)

        self.model = Model(inputs=inputs, outputs=self.out)

        self.fitted = False
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self, X, y, X_test, y_test, epochs=10):
        self.model.fit(
            x=X,
            y=y,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            epochs=epochs,
        )
        self.fitted = True

    def predict_proba(self, batch):
        assert self.fitted
        return self.model.predict(x=batch, batch_size=self.batch_size)

    def predict(self, batch):
        return np.round(self.predict_proba(batch))

    def save(self, filename):
        print(f"=== Saving Fine Tuned Bert Model : {filename} ===")
        self.model.save_weights(filename)

    def load(self, filename):
        print("=== Loading Fine Tuned Bert Model : {filename} === ")
        self.model.load_weights(filename)
        self.fitted = True
