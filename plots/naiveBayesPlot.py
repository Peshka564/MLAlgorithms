import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from plots.boundaryPlot import BoundaryPlot
from algorithms.naiveBayes import NaiveBayes

class NBPlot(BoundaryPlot):
    def __init__(self, range, num_labels, dict_size):
        super().__init__(range)

    def train(self):
        ...

    def predict(self, grid_points):
        ...