import numpy as np

from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

class LSTMClf:
    def __init__(self, batch_size, length, embedding_matrix=None, loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=3e-5),metrics=['accuracy']):

        self.batch_size = batch_size   
        self.input = Input(shape=(length,), name="input_ids", dtype="int32")
        if embedding_matrix is None :
            self.embedding = Embedding(
                1000, output_dim=256, input_length=length, trainable=True
            )(self.input)
        else :
            self.embedding = Embedding(
                np.shape(embedding_matrix)[0], output_dim=np.shape(embedding_matrix)[1], input_length=length, trainable=False
            )(self.input)
        self.lstm1 = LSTM(128)(self.embedding)
        self.dropout = Dropout(0.2)(self.lstm1)
        self.output = Dense(1, activation="sigmoid")(self.dropout)
        self.model = Model(inputs=self.input, outputs=self.output)

        self.fitted = False
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self, X, y, X_test, y_test, epochs=10):
        self.model.fit(x=X, y=y, batch_size=self.batch_size, validation_data=(X_test, y_test), epochs=epochs)
        self.fitted = True        

    def predict_proba(self, batch):
        assert self.fitted
        return self.model.predict(x=batch, batch_size=self.batch_size) 

    def predict(self, batch):
        return np.round(self.predict_proba(batch))

    def save(self, filename):
        pass 

    def load(self, filename):
        pass  

