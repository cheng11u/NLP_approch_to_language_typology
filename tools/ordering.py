from grewpy import Corpus, Request
import numpy as np


def compute_ordered_distributions(
    request: Request, corpora: list[Corpus]
) -> tuple[list[list[float]], list[str], list[dict[str, int]]]:
    clustering_parameter = ["#".join(request.named_entities()["nodes"])]
    corpora_counts = [
        corpus.count(request, clustering_parameter=clustering_parameter)
        for corpus in corpora
    ]
    orderings = list(
        set([key for count in corpora_counts for key in count.keys()]))
    vectors = [[
        corpus_count[ordering] if ordering in corpus_count.keys() else 0
        for ordering in orderings
    ] for corpus_count in corpora_counts]
    vectors = [vector / np.sum(vector) for vector in vectors]
    return vectors, orderings, corpora_counts


def extract_ordered_examples(
    request: Request, corpora: list[Corpus]
) -> tuple[list[list[float]], list[str], list[dict[str, int]]]:
    clustering_parameter = ["#".join(request.named_entities()["nodes"])]
    corpora_examples = [
        corpus.search(request, clustering_parameter=clustering_parameter)
        for corpus in corpora
    ]
    return corpora_examples
