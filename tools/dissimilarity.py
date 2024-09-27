import numpy as np
from abc import ABC, abstractmethod
from forest.benchmarking.distance_measures import total_variation_distance

class DissimilarityMeasure(ABC):

    def __init__():
        pass

    @abstractmethod
    def compare(self, v1: list[float], v2: list[float]) -> float:
        assert (len(v1) == len(v2))

    def distanceMatrix(self, vectors: list[list[float]]) -> list[list[float]]:
        return [[self.compare(v1, v2) for v2 in vectors] for v1 in vectors]


class CosineSimilarity(DissimilarityMeasure):

    def __init__(self):
        pass

    def compare(self, v1: list[float], v2: list[float]) -> float:
        super().compare(v1, v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class TotalVariationDistance(DissimilarityMeasure):

    def __init__(self):
        pass

    def compare(self, v1: list[float], v2: list[float]) -> float:
        super().compare(v1, v2)
        #return total_variation_distance(np.array(v1), np.array(v2))
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        diff = v2_array - v1_array
        diff_abs = np.abs(diff)
        sum_diff = np.sum(diff_abs)
        return sum_diff / 2