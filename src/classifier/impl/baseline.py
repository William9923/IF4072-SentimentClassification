import numpy as np

from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from src.classifier.interface import IClassifier


class LSTMClf(IClassifier):
    def __init__(
        self,
        batch_size,
        length,
        epochs,
        embedding_matrix,
        embedding_matrix_shape,
        loss,
        optimizer,
        metrics,
    ):

        self.batch_size = batch_size
        self.length = length
        embedding_matrix_shape
        self.input = Input(shape=(length,), name="input_ids", dtype="int32")
        if embedding_matrix is None:
            self.embedding = Embedding(
                embedding_matrix_shape[0], output_dim=embedding_matrix_shape[1], input_length=length, trainable=True
            )(self.input)
        else:
            self.embedding = Embedding(
                np.shape(embedding_matrix)[0],
                output_dim=np.shape(embedding_matrix)[1],
                input_length=length,
                trainable=False,
            )(self.input)
        self.lstm1 = LSTM(128)(self.embedding)
        self.dropout = Dropout(0.2)(self.lstm1)
        self.output = Dense(1, activation="sigmoid")(self.dropout)
        self.model = Model(inputs=self.input, outputs=self.output)

        self.fitted = False

        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        self.epochs = epochs

    def set_embedding_matrix(self, embedding_matrix):
        self.embedding = Embedding(
            np.shape(embedding_matrix)[0],
            output_dim=np.shape(embedding_matrix)[1],
            input_length=self.length,
            trainable=False,
        )(self.input)
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
        )

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
        print(f"=== Saving Baseline (LSTM) Model : {filename} ===")
        self.model.save_weights(filename)

    def load(self, filename):
        print("=== Loading Baseline (LSTM) Model : {filename} === ")
        self.model.load_weights(filename)
        self.fitted = True
