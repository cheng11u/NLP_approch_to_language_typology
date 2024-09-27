import grewpy
from grewpy import Request

from typing import Callable
import itertools

grewpy.set_config("ud")


class OrderedRequest:

    def __init__(self, request_str: str) -> None:
        self._orderings = [
            ";".join([
                permutation[i] + "<<" + permutation[i + 1]
                for i in range(len(permutation) - 1)
            ]) for permutation in itertools.permutations(
                Request(request_str).named_entities()["nodes"])
        ]

        self._requests = {
            ordering: Request(request_str).with_(ordering)
            for ordering in self._orderings
        }
        self._requests["#"] = Request(request_str)

    def distribution(self, corpus: grewpy.Corpus) -> tuple[dict, dict]:
        ordering_counts = {
            ordering: corpus.count(request)
            for ordering, request in self._requests.items()
        }

        return {
            ordering: count / ordering_counts["#"]
            for ordering, count in ordering_counts.items()
        }, ordering_counts
