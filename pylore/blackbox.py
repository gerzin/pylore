from abc import abstractmethod, ABC


class AbstractBlackBoxWrapper(ABC):

    @abstractmethod
    def predict(self):
        pass
