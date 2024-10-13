import matplotlib.pyplot as plt
from grewpy import Corpus, Request
from tools.ordering import compute_ordered_distributions
from tools.charts.heatmap import HeatMap
from tools.dissimilarity import CosineSimilarity, TotalVariationDistance
from tools.search_corpora import select_by_language  
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
#corpora_names = [
#    "UD_French-GSD", "UD_German-GSD", "UD_English-GUM", "UD_Thai-PUD",
#    "UD_Japanese-PUD", "UD_Chinese-PUD", "UD_Swedish-PUD", "UD_Italian-ISDT"
#]

#corpora_names = [
#    "UD_French-FQB", "UD_French-GSD", "UD_French-PUD", "UD_French-Sequoia",
#    "UD_French-Rhapsodie", "UD_French-ParTUT"
#]
#corpora_path = [
#    "data/ud-treebanks-v2.14/" + corpus_name for corpus_name in corpora_names
#]

corpora_path = select_by_language("UD", "data/ud-treebanks-v2.14", "French", 1000)
corpora_names = [path.split("/")[-1] for path in corpora_path]
request = Request(pattern_str)
corpora = [Corpus(path) for path in corpora_path]
distance = CosineSimilarity()
distributions, labels, counts = compute_ordered_distributions(request, corpora)

c = CosineSimilarity()
distance_matrix = c.distanceMatrix(distributions)
for i, row in enumerate(distance_matrix):
    for j, value in enumerate(distance_matrix[i]):
        distance_matrix[i][j] = value


print("Ordering counts")
for i, corpus_name in enumerate(corpora_names):
    print(f"------- Corpus {corpus_name} --------")
    for k, v in counts[i].items():
        print(f"{k} : {v}")

hm = HeatMap()
fig, ax = hm.exportFigure(distance_matrix, corpora_names)
ax.set_title("Comparision of AN distributions among French UD corpora")
plt.show()
