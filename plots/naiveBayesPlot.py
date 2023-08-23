import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from plots.boundaryPlot import BoundaryPlot
from algorithms.naiveBayes import GaussianNB, LinearGaussianNB
#from sklearn.naive_bayes import GaussianNB

class NBPlot(BoundaryPlot):
    def __init__(self, range):
        super().__init__(range)

    def train(self):
        blues = np.array([(i, j, 0) for i, j in zip(self.blue_x, self.blue_y)])
        reds = np.array([(i, j, 1) for i, j in zip(self.red_x, self.red_y)])
        #blues = np.array([(i, j) for i, j in zip(self.blue_x, self.blue_y)])
        #reds = np.array([(i, j) for i, j in zip(self.red_x, self.red_y)])
        training_points = np.concatenate((blues, reds), axis=0)
        #training_labels = np.concatenate((np.full(shape=(len(blues), 1), fill_value=-1), np.full(shape=(len(reds), 1), fill_value=1)), axis=0)

        self.gnb = LinearGaussianNB()
        self.gnb.fit(training_points, 2)
        #self.gnb.fit(training_points, training_labels)
        

    def predict(self, grid_points):
        predictions = np.empty(len(grid_points))
        for grid_index in range(len(grid_points)):
            predictions[grid_index] = self.gnb.predict(grid_points[grid_index])
            #predictions[grid_index] = self.gnb.predict(np.array([grid_points[grid_index]]))
        return predictions