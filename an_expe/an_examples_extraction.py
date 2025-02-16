import argparse
import matplotlib.pyplot as plt
from grewpy import Corpus, Request
from tools.ordering import compute_ordered_distributions, extract_ordered_examples
from tools.charts.heatmap import HeatMap
from tools.dissimilarity import CosineSimilarity, TotalVariationDistance
import grewpy
import json

def extract_adj_noun_examples(corpora_names):
    """Extract the ADJ/NOUN examples from specified UD_CORPORA"""
    grewpy.set_config("ud")  # ud or basic

    pattern_str = """
    pattern {
        N [upos="NOUN"];
        A [upos="ADJ"];
        N -[amod]-> A;
    }
    """

    corpora_path = ["data/ud-treebanks-v2.14/" + corpus_name for corpus_name in corpora_names]
    request = Request(pattern_str)
    corpora = [Corpus(path) for path in corpora_path]
    examples = extract_ordered_examples(request, corpora)

    examples_dict = {
        "corpora": [{
            "corpus": corpus_name,
            "samples": {
                key: [{
                    "A": corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]["A"]]["lemma"],
                    "N": corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]["N"]]["lemma"]
                } for sentence in sentences]
                for key, sentences in examples[i].items()
            }
        } for i, corpus_name in enumerate(corpora_names)]
    }

    print(json.dumps(examples_dict, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ADJ/NOUN examples from UD corpora")
    parser.add_argument("corpora_names", nargs='+', help="List of UD corpora names")
    args = parser.parse_args()

    extract_adj_noun_examples(args.corpora_names)
