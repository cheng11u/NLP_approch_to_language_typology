import tools.charts
from tools.dissimilarity import CosineSimilarity as cd
from tools.charts.heatmap import HeatMap

c = cd()
vectors = [[0, 1, 2], [0, 2, 4], [0, 0, 1], [0, 0.1, 1]]
distances = c.distanceMatrix(vectors)
labels = list("ABCD")

hm = HeatMap ()
hm.exportFigure (distances, labels)

