from abc import ABC, abstractmethod

class IClassifier(ABC):
    
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
    def summary(self):
        pass 

    @abstractmethod
    def save(self):
        pass 

    @abstractmethod
    def load(self):
        pass 
    
