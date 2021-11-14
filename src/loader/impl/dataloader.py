import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.loader.interface import ILoader

class DataLoader(ILoader):
    def __init__(self, target, sample_size, sampling, train_file_path, test_file_path, val_split):
        
        assert val_split <= 1
        
        self.target = target
        self.sample_size=sample_size
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.val_size = val_split
        self.sampling = sampling
        self.encoder = {
            "neutral" : 0,
            "positive": 1,
            "negative": 2,
        }
        self.reverse_encoder = {}
        for key, val in self.encoder.items():
            self.reverse_encoder[val] = key

    def load(self):
        train = pd.read_csv(self.train_file_path)
        test = pd.read_csv(self.test_file_path)

        assert len(train) >= self.sample_size
        train = train.dropna(subset=['review'])
        train[self.target] = train[self.target].astype(str).map(self.encoder)
        test = test.dropna(subset=['review'])
        test[self.target] = test[self.target].astype(str).map(self.encoder)

        X = train.drop([self.target], axis = 1)
        y = train[self.target]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, stratify=y)
        self.train = (X_train, y_train)
        self.val = (X_val, y_val)

        X_test = test.drop([self.target], axis = 1)
        y_test = test[self.target]
        self.test = (X_test, y_test)

    def get_train_data(self):
        if self.train is None:
            self.load()

        if self.sampling:
            return (self.train[0][:self.sample_size], self.train[1][:self.sample_size])
        return self.train

    def get_val_data(self):
        if self.val is None:
            self.load()
        
        if self.sampling:
            return (self.val[0][:self.sample_size], self.val[1][:self.sample_size])
        return self.val

    def get_test_data(self):
        if self.test is None:
            self.load()

        if self.sampling:
            return (self.test[0][:self.sample_size], self.test[1][:self.sample_size])
        return self.test

    def reverse_labels(self, batch):
        assert self.encoder is not None
        reversed_batch = []
        for item in batch: 
            reversed_batch.append(self.reverse_encoder.get(item))
        return reversed_batch



    


