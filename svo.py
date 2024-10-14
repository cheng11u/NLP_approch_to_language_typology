import matplotlib.pyplot as plt
from grewpy import Corpus, Request
from tools.ordering import compute_ordered_distributions
from tools.charts.heatmap import HeatMap
from tools.dissimilarity import CosineSimilarity
from tools.rewriting import add_implicit_subject
import grewpy
"""Computes and compare the ADJ/NOUN distribution of different UD_CORPORA"""

grewpy.set_config("ud")  # ud or basic

pattern_str = "pattern {V[upos=VERB];V-[nsubj|isubj]->S; V-[obj]->O}"

# corpora_names = [
#    "UD_French-GSD", "UD_German-GSD", "UD_English-GUM", "UD_Thai-PUD",
#    "UD_Japanese-PUD", "UD_Chinese-PUD", "UD_Swedish-PUD", "UD_Italian-ISDT"
# ]

corpora_names = [
    "UD_French-FQB", "UD_French-GSD", "UD_French-PUD", "UD_French-Sequoia",
    "UD_French-Rhapsodie", "UD_French-ParTUT"
]
corpora_path = [
    "data/ud-treebanks-v2.14/" + corpus_name for corpus_name in corpora_names
]
request = Request(pattern_str)
corpora = [Corpus(path) for path in corpora_path]
corpora = [add_implicit_subject(corpus) for corpus in corpora]
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
ax.set_title("")
#ax.set_title("Comparision of SVO distributions among French UD banks")
#plt.show()
fig.savefig('svo.png', bbox_inches='tight', dpi=400)
