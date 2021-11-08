from abc import ABC, abstractmethod

class IPreprocessor(ABC):
    
    @abstractmethod
    def preprocess(self):
        pass 

    @abstractmethod
    def available_component(self):
        pass 