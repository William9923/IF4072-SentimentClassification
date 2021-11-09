from abc import ABC, abstractmethod

class IBoWFeatureExtractor(ABC):
    
    @abstractmethod
    def fit(self):
        pass 

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod 
    def fit_transform(self):
        pass

    @abstractmethod
    def save(self):
        pass 

    @abstractmethod
    def load(self):
        pass

class IW2VFeatureExtractor(ABC):
    
    @abstractmethod
    def train(self):
        pass 

    @abstractmethod 
    def tokenize(self, X):
        pass

    @abstractmethod
    def get_embedding_matrix(self):
        pass

    @abstractmethod
    def save(self):
        pass 

    @abstractmethod
    def load(self):
        pass