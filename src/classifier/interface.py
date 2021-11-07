from abc import ABC, abstractmethod

class IShallowMLClassifier(ABC):
    
    @abstractmethod
    def fit(self):
        pass 

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict_proba(self):
        pass 

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def load(self):
        pass 

    @abstractmethod
    def save(self):
        pass

class IDeepMLClassifier(ABC):

    @abstractmethod
    def compile():
        pass

    @abstractmethod
    def summary():
        pass

    @abstractmethod
    def fit(self):
        pass 

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict_proba(self):
        pass 

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def load(self):
        pass 

    @abstractmethod
    def save(self):
        pass