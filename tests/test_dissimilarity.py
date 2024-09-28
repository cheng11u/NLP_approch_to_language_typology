import unittest
import inspect
import tools.dissimilarity
import numpy as np


class Dissimilarity(unittest.TestCase):

    def setUp(self):
        self.distribution = np.random.rand(16, 10)

        # Making sure the two first distributions
        # are different so that we can check the distance
        # between both is non zero
        self.distribution[1, 0] = self.distribution[0, 0] + 1

        self.distribution = self.distribution / np.sum(self.distribution,
                                                       axis=1)[:, np.newaxis]

    def test_dissimilarity_measures(self):
        for name, obj in inspect.getmembers(tools.dissimilarity):
            if name != "DissimilarityMeasure" and inspect.isclass(
                    obj) and issubclass(
                        obj, tools.dissimilarity.DissimilarityMeasure):
                print(f"Testing {name}")
                try:

                    score = obj().compare(self.distribution[0],
                                          self.distribution[1])
                    score_ = obj().compare(self.distribution[0],
                                           self.distribution[0])
                    distanceMatrix = obj().distanceMatrix(self.distribution)
                except Exception as e:
                    self.fail(f"Error testing class {name} : {e} ")
                    if name == "CosineSimilarity":
                        assert (score < 1)
                        assert (score_ == 1)
                    else:
                        assert (score > 0)
                        assert (score_ == 0)
