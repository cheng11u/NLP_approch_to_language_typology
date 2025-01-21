import numpy as np
import fasttext
import matplotlib.pyplot as plt
from grewpy import Corpus, Request
from tools.ordering import compute_ordered_distributions, extract_ordered_examples
from tools.charts.heatmap import HeatMap
from tools.dissimilarity import CosineSimilarity, TotalVariationDistance
import grewpy
import json
"""Extract the ADJ/NOUN examples of different UD_CORPORA"""

grewpy.set_config("ud")  # ud or basic

pattern_str = """
pattern {
    N [upos="NOUN"];
    A [upos="ADJ"];
    N -[amod]-> A;
}
"""
corpora_names = [
    "UD_French-FQB", "UD_French-GSD", "UD_French-PUD", "UD_French-Sequoia",
    "UD_French-Rhapsodie", "UD_French-ParTUT"
]

corpora_path = [
    "data/ud-treebanks-v2.14/" + corpus_name for corpus_name in corpora_names
]
request = Request(pattern_str)
corpora = [Corpus(path) for path in corpora_path]
examples = extract_ordered_examples(request, corpora)
fasttext_model = fasttext.load_model("models/cc.fr.300.bin")
filter_words = [
    "antidepresseur", "obstétricien", "pharmacie", "pharmacien", "patient",
    "hopital", "medecin", "medicament", "veterinaire", "cardiologue"
]
filter_embeddings = [fasttext_model[word] for word in filter_words]


def compare(v1: list[float], v2: list[float]) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def filter(word, threshold=0.5):
    e = fasttext_model[word]
    dist = max([compare(e, e_) for e_ in filter_embeddings])
    if dist > threshold:
        return False
    return True


examples_dict = {
    "corpora": [{
        "corpus": corpus_name,
        "samples": {
            key: [{
                "A":
                corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]
                                                ["A"]]["lemma"],
                "N":
                corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]
                                                ["N"]]["lemma"]
            } for sentence in sentences
                  if not filter(corpora[i][sentence["sent_id"]][
                      sentence["matching"]["nodes"]["N"]]["lemma"])]
            for key, sentences in examples[i].items()
        }
    } for i, corpus_name in enumerate(corpora_names)]
}
print(json.dumps(examples_dict))
