import numpy as np
from abc import ABC, abstractmethod


class DissimilarityMeasure(ABC):
    """
    An abstract class used to represent a dissimilarity measure

    Methods
    -------
    compare (v1, v2)
        computes the distance between two distributions

    distanceMatrix(vectors)
        computes the pairwise distance matrix
    """

    def __init__():
        pass

    @abstractmethod
    def compare(self, v1: list[float], v2: list[float]) -> float:
        """ Computes the distance between two distributions
            Parameters
            ----------
            v1, v2 : vectors representing two probability distributions
        """
        assert (len(v1) == len(v2))

    def distanceMatrix(self, vectors: list[list[float]]) -> list[list[float]]:
        """ Computes the pairwise distance matrix
            Parameters
            ---------
            vectors : a list of distribution vectors
         """
        return [[self.compare(v1, v2) for v2 in vectors] for v1 in vectors]


class CosineSimilarity(DissimilarityMeasure):
    """ A class implementing cosine similarity for probability distributions"""

    def __init__(self):
        pass

    def compare(self, v1: list[float], v2: list[float]) -> float:
        super().compare(v1, v2)
        return float(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


class TotalVariationDistance(DissimilarityMeasure):
    """ A class implementing total variation distance for
    probability distributions"""

    def __init__(self):
        pass

    def compare(self, v1: list[float], v2: list[float]) -> float:
        super().compare(v1, v2)
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        diff = v2_array - v1_array
        diff_abs = np.abs(diff)
        sum_diff = np.sum(diff_abs)
        return sum_diff / 2
