from abc import ABC, abstractmethod
import matplotlib
import matplotlib.figure as Figure
from typing import Any

class Chart(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def exportFigure(
            self, *args:
        Any) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        pass
