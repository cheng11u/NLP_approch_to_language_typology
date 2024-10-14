import matplotlib.pyplot as plt
from grewpy import Corpus, Request
from tools.ordering import compute_ordered_distributions
from tools.charts.heatmap import HeatMap
from tools.dissimilarity import CosineSimilarity, TotalVariationDistance
import grewpy
"""Computes and compare the ADJ/NOUN distribution of different UD_CORPORA"""

grewpy.set_config("ud")  # ud or basic

pattern_str = '''
pattern {
    N [upos="NOUN"];
    A [upos="ADJ"];
    N -[amod]-> A;
}
'''
corpora_names = [
    "UD_French-GSD", "UD_Italian-PUD", "UD_Spanish-GSD", "UD_Latin-ITTB",
    "UD_German-GSD", "UD_English-PUD", "UD_Dutch-Alpino"
]

#corpora_names = [
#    "UD_French-FQB", "UD_French-GSD", "UD_French-PUD", "UD_French-Sequoia",
#    "UD_French-Rhapsodie", "UD_French-ParTUT"
#]
corpora_path = [
    "data/ud-treebanks-v2.14/" + corpus_name for corpus_name in corpora_names
]
request = Request(pattern_str)
corpora = [Corpus(path) for path in corpora_path]
distance = CosineSimilarity()
distributions, labels, counts = compute_ordered_distributions(request, corpora)

c = CosineSimilarity()
distance_matrix = c.distanceMatrix(distributions)

print("Ordering counts")
for i, corpus_name in enumerate(corpora_names):
    print(f"------- Corpus {corpus_name} --------")
    for k, v in counts[i].items():
        print(f"{k} : {v}")

hm = HeatMap()
fig, ax = hm.exportFigure(distance_matrix, corpora_names)
ax.set_title(
    ""
)
fig.savefig('an.png', bbox_inches='tight', dpi=400)
