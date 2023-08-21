from boundaryPlot import BoundaryPlot
from naiveBayes import NaiveBayes

class NBPlot(BoundaryPlot):
    def __init__(self, range, num_labels, dict_size):
        super().__init__(range)

    def train(self):
        ...

    def predict(self, grid_points):
        ...