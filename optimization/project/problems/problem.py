from abc import ABC, abstractmethod

class Problem(ABC):

    @abstractmethod
    def fun(self): pass

    @abstractmethod
    def grad(self): pass
