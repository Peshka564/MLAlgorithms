import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from algorithms.perceptron import Perceptron
from plots.boundaryPlot import BoundaryPlot
import numpy as np

class PerceptronPlot(BoundaryPlot):
    def __init__(self, range):
        super().__init__(range)

    def train(self):
        # Prepare data
        blues = np.array([(i, j) for i, j in zip(self.blue_x, self.blue_y)])
        reds = np.array([(i, j) for i, j in zip(self.red_x, self.red_y)])
        training_points = np.concatenate((blues, reds), axis=0)
        training_labels = np.concatenate((np.full(shape=(len(blues), 1), fill_value=-1), np.full(shape=(len(reds), 1), fill_value=1)), axis=0)
        
        # Train
        self.perceptron = Perceptron()
        self.perceptron.fit(training_points, training_labels)
        
        print(self.perceptron.get_parameters())
    
    def predict(self, grid_points):
        grid_points = np.concatenate((grid_points, np.full(shape=(len(grid_points), 1), fill_value=1)), axis=1)
        return self.perceptron.predict(grid_points)