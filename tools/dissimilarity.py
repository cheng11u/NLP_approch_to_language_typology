import numpy as np
from abc import ABC, abstractmethod


class DissimilarityMeasure(ABC):

    def __init__():
        pass

    @abstractmethod
    def compare(self, v1: list[float], v2: list[float]) -> float:
        return 0

    def distanceMatrix(self, vectors: list[list[float]]) -> list[list[float]]:
        return [[self.compare(v1, v2) for v2 in vectors] for v1 in vectors]


class CosineDissimilarity(DissimilarityMeasure):

    def __init__(self):
        pass

    def compare(self, v1: list[float], v2: list[float]) -> float:
        assert (len(v1) == len(v2))
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
