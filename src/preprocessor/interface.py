from abc import ABC, abstractmethod

class IPreprocessor(ABC):
    
    @abstractmethod
    def preprocess(self):
        pass 