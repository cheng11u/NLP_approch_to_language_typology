from tools.charts.chart import Chart
import matplotlib.pyplot as plt
import numpy as np


class HeatMap(Chart):

    def __init__(self):
        pass

    def exportFigure(self, distanceMatrix: list[list[float]],
                     labels: list[str]) :
        fig, ax = plt.subplots()
        im = plt.imshow(1 - np.array(distanceMatrix), cmap="viridis")
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.tick_params(top=True,
                       labeltop=True,
                       bottom=False,
                       labelbottom=False)

        plt.setp(ax.get_xticklabels(),
                 rotation=-45,
                 ha="right",
                 rotation_mode="anchor")

        for i in range(len(distanceMatrix)):
            for j in range(len(distanceMatrix[i])):
                text = ax.text(j,
                               i,
                               f"{distanceMatrix[i][j]:.4f}",
                               ha="center",
                               va="center",
                               color="w")

        ax.set_title("Change me")
        fig.tight_layout()
        return fig, ax
