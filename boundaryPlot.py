from pointPlot import PointPlot
import numpy as np
from abc import ABC, abstractmethod

class BoundaryPlot(PointPlot, ABC):
    def __init__(self, range):
        super().__init__(range)

    def updateAfterPlot(self):
        x = np.linspace(self.range[0], self.range[1], (self.range[1] - self.range[0]) * 10)
        y = np.linspace(self.range[0], self.range[1], (self.range[1] - self.range[0]) * 10)
        self.train()
        x, y = np.meshgrid(x, y)
        grid_points = np.array([[point_x, point_y] for point_x, point_y in zip(x.flatten(), y.flatten())])

        p = self.predict(grid_points).reshape(x.shape)
        p[p == 0] = -1
        
        self.ax.contourf(x, y, p, levels=[-1, 0, 1], colors=("cyan", "salmon"), zorder=0)

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def predict(self, grid_points):
        ...
        