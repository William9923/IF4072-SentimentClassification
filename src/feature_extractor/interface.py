from abc import ABC, abstractmethod

class IFeatureExtractor(ABC):
    
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