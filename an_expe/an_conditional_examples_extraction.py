import argparse
import numpy as np
import fasttext
import json
from tqdm import tqdm
from grewpy import Corpus, Request
from tools.ordering import extract_ordered_examples
import grewpy

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Extract ADJ/NOUN examples from UD corpora with optional filtering.")
parser.add_argument("corpora", nargs="+", help="List of UD corpora names to process")
parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for filtering words (default: 0.2)")
parser.add_argument("--filter-words", type=str, nargs="*", default=[
    "antidepresseur", "obstétricien", "pharmacie", "pharmacien", "patient",
    "hopital", "medecin", "medicament", "veterinaire", "cardiologue"
], help="List of words to filter out")
parser.add_argument("--output-positives", type=str, default="data/positives.json",
                    help="Filename for positive examples (default: data/positives.json)")
parser.add_argument("--output-non-positives", type=str, default="data/non_positives.json",
                    help="Filename for non-positive examples (default: data/non_positives.json)")

args = parser.parse_args()
corpora_names = args.corpora
threshold = args.threshold
filter_words = args.filter_words
output_positives = args.output_positives
output_non_positives = args.output_non_positives

# Set grewpy config
grewpy.set_config("ud")  # ud or basic

# Define pattern
pattern_str = """
pattern {
    N [upos="NOUN"];
    A [upos="ADJ"];
    N -[amod]-> A;
}
"""
corpora_path = [f"data/ud-treebanks/{corpus_name}" for corpus_name in corpora_names]
request = Request(pattern_str)
corpora = [Corpus(path) for path in corpora_path]
examples = extract_ordered_examples(request, corpora)

# Load FastText model
fasttext_model = fasttext.load_model("models/cc.fr.300.bin")

# Compute embeddings for filter words
filter_embeddings = [fasttext_model[word] for word in filter_words]


def compare(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def filter(word):
    """Check if a word should be filtered based on similarity to specified filter words."""
    e = fasttext_model[word]
    dist = max([compare(e, e_) for e_ in filter_embeddings])
    return dist <= threshold


# Generate examples with and without filtering
examples_non_dict = {
    "corpora": [{
        "corpus": corpus_name,
        "samples": {
            key: [{
                "A": corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]["A"]]["lemma"],
                "N": corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]["N"]]["lemma"]
            } for sentence in sentences if filter(corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]["A"]]["lemma"])]
            for key, sentences in examples[i].items()
        }
    } for i, corpus_name in tqdm(enumerate(corpora_names))]
}

examples_dict = {
    "corpora": [{
        "corpus": corpus_name,
        "samples": {
            key: [{
                "A": corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]["A"]]["lemma"],
                "N": corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]["N"]]["lemma"]
            } for sentence in sentences if not filter(corpora[i][sentence["sent_id"]][sentence["matching"]["nodes"]["A"]]["lemma"])]
            for key, sentences in examples[i].items()
        }
    } for i, corpus_name in tqdm(enumerate(corpora_names))]
}

# Save output to specified JSON files
with open(output_positives, "w") as f:
    json.dump(examples_dict, f, ensure_ascii=False)

with open(output_non_positives, "w") as f:
    json.dump(examples_non_dict, f, ensure_ascii=False)

print(f"Saved positive examples to {output_positives}")
print(f"Saved non-positive examples to {output_non_positives}")

