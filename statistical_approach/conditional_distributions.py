import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from grewpy import Corpus, Request
from tools.ordering import compute_ordered_distributions, extract_ordered_examples
from tools.charts.heatmap import HeatMap
from tools.dissimilarity import CosineSimilarity, TotalVariationDistance, CosineSimilarity
import grewpy
import json
import scipy.stats


grewpy.set_config("ud")  # ud or basic

pattern_str = """
pattern {
    N [upos="NOUN"];
    A [upos="ADJ"];
    N -[amod]-> A;
    }
            """
# corpora_names = [
#    "UD_French-GSD", "UD_German-GSD", "UD_English-GUM",
#    "UD_Thai-PUD",
#    "UD_Japanese-PUD", "UD_Chinese-PUD", "UD_Swedish-PUD",
#    "UD_Italian-ISDT"
# ]

corpora_names = [
    "UD_French-FQB", "UD_French-GSD", "UD_French-PUD", "UD_French-Sequoia",
    "UD_French-Rhapsodie", "UD_French-ParTUT"
]
corpora_path = [
    "data/ud-treebanks/" + corpus_name for corpus_name in corpora_names
]
request = Request(pattern_str)
corpora = [Corpus(path) for path in corpora_path]
distance = TotalVariationDistance()
similarity = CosineSimilarity()
examples = extract_ordered_examples(request, corpora)
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
            } for sentence in sentences]
            for key, sentences in examples[i].items()
        }
    } for i, corpus_name in enumerate(corpora_names)]
}

variables_1 = ['N << A', 'A << N']
variables_2 = ['A', 'N']
distributions = {}
for v1 in variables_1:
    for v2 in variables_2:
        distributions[v2 + "|" + v1] = Counter([
            sample[v2] for corpus in examples_dict["corpora"]
            for sample in corpus["samples"][v1]
        ])

original_distributions = distributions

#print(sorted(distributions['A|A << N'].items(), key=lambda x:x[1])[-10:])
#print(sorted(distributions['N|A << N'].items(), key=lambda x:x[1])[-10:])
#print(sorted(distributions['A|N << A'].items(), key=lambda x:x[1])[-10:])
#print(sorted(distributions['N|N << A'].items(), key=lambda x:x[1])[-10:])
for threshold in range(0,51,10):
    print("Threshold",threshold)
    distributions = {
        outer_key: {
            inner_key: value
            for inner_key, value in inner_dict.items() if value>threshold
        }
        for outer_key, inner_dict in original_distributions.items()
    }

    distributions_normalized = {
        outer_key: {
            inner_key: value / sum(inner_dict.values())
            for inner_key, value in inner_dict.items()
        }
        for outer_key, inner_dict in distributions.items()
    }
    for v2 in variables_2:
        print(f"Comparison of {v2} distributions")
        d0 = distributions_normalized[v2 + "|" + variables_1[0]]
        d1 = distributions_normalized[v2 + "|" + variables_1[1]]
        keys = set(d0.keys()) | set(d1.keys())
        vector1 = [d0[key] if key in d0.keys() else 0 for key in keys]
        vector2 = [d1[key] if key in d1.keys() else 0 for key in keys]
        print("TVD",distance.compare(vector1, vector2))
        print("Cosine sim",similarity.compare(vector1, vector2))
        #d0 = distributions[v2 + "|" + variables_1[0]]
        #d1 = distributions[v2 + "|" + variables_1[1]]
        #keys = set(d0.keys()) | set(d1.keys())
        #vector1 = [d0[key] if key in d0.keys() else 0 for key in keys]
        #vector2 = [d1[key] if key in d1.keys() else 0 for key in keys]
        #contingency = [vector1,vector2]
        #print(scipy.stats.chi2_contingency(contingency).pvalue)
