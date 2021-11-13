import numpy as np

from tensorflow.keras.layers import (
    Input,
    Dropout,
    Dense,
    LSTM,
)
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from transformers import TFDistilBertModel

from src.classifier.interface import IClassifier


class FineTuneBertClf(IClassifier):
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
        self.lstm = LSTM(128)(self.embedding)
        # self.fcn1 = Dense(length, activation="relu")(self.embedding[:, 0, :])
        self.fcn1 = Dense(64, activation="relu")(self.lstm)
        self.fcn2 = Dropout(0.5)(self.fcn1)
        self.out = Dense(1, activation="sigmoid")(self.fcn2)

        self.model = Model(inputs=inputs, outputs=self.out)

        self.fitted = False
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.epochs = epochs

    def train(self, X, y, X_test, y_test):
        self.model.fit(
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
        return np.round(self.predict_proba(batch))

    def save(self, filename):
        formatted_filename_model = f"{filename}.hd5"
        formatted_filename_weight = f"{filename}.h5"

        print(f"=== Saving Fine Tuned Bert Model : {filename} ===")
        self.model.save_weights(formatted_filename_weight)
        json_rpr = self.model.to_json()
        with open(formatted_filename_model, 'w') as f:
            f.write(json_rpr)

    def load(self, filename):
        formatted_filename_model = f"{filename}.hd5"
        formatted_filename_weight = f"{filename}.h5"

        print("=== Loading Fine Tuned Bert Model : {filename} === ")
        with open(formatted_filename_model, 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(formatted_filename_weight)
        self.fitted = True
