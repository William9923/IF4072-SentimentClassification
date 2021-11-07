from abc import ABC, abstractmethod

class ILoader(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_train_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

    @abstractmethod
    def shuffle(self):
        pass

    @abstractmethod
    def reverse_labels(self, batch):
        pass
