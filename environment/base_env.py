from abc import abstractmethod, ABC


class IBaseEnv(ABC):

    @abstractmethod
    def fitness_function(self, individual):
        ...

    @abstractmethod
    def constraints(self, individual):
        ...
