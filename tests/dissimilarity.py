from tools.dissimilarity import CosineDissimilarity

c = CosineDissimilarity()
print(c.compare([1, 2,3], [3, 4, 5]))
print(c.distanceMatrix ([[1,2,3],[2,3,4],[2,4,6],[4,6,8]]))
print(c.compare([0,1],[1,0]))
print(c.compare([1,0],[1/2,0.875]))


