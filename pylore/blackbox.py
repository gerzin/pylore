from abc import abstractmethod, ABC


class AbstractBlackBoxWrapper(ABC):
    """
    Interface for using a black box with LORE.

    No matter the framework (Pytorch, Keras, Tensorflow) just create a class which wraps the model
    and exposes a `predict` method in order to obtain predictions on instances.
    """

    @abstractmethod
    def predict(self, x):
        pass
