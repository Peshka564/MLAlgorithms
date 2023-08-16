from classificationPlot import ClassificationPlot
from knn import knn, knn_eval
import matplotlib.pyplot as plt
import numpy as np

class KNN_Plotter(ClassificationPlot):

    def __init__(self, k, range):
        if k == 0: raise Exception("Invalid k")
        super().__init__(range)
        self.firstPrediction = True
        self.k = k

    def updateAfterPlot(self):
        self.title = "Click to plot test point"
        self.color = "black"
        self.ax.set_title(self.title)
    
    # This is knn with a single testing point
    def classify(self, testing_point):
        
        # deal with excess lines and points for previous testing point
        if not self.firstPrediction:
            for i in range(0, len(self.temporaries)):
                self.temporaries[i].remove()
            self.temporaries = []
        else: self.temporaries = []

        # Preparing data to feed to knn
        training_x = np.hstack((self.blue_x, self.red_x))
        training_y = np.hstack((self.blue_y, self.red_y))
        training_data = np.stack((training_x, training_y), axis=1)
        labels = np.hstack((np.array([0] * len(self.blue_x)), np.array([1] * len(self.red_x))))
        label, indices = knn(training_data, labels, np.array([testing_point]), self.k, 2)

        # Drawing prediction
        for index in indices[0]:
            # connect points to visualize nearest neighbours
            self.temporaries.append(self.ax.plot([training_data[index][0], testing_point[0]], [training_data[index][1], testing_point[1]], color="green")[0])
        self.temporaries.append(self.ax.scatter(testing_point[0], testing_point[1], color=self.color))

        return label[0]
    
    def additional_actions(self, testing_point):
        # Classify plotted test point
        if self.k > len(self.blue_x) + len(self.red_x): raise Exception("Too little points")

        self.title = f"The labels is: {'Blue' if self.classify(testing_point) == 0 else 'Red'}"
        self.ax.set_title(self.title)
        self.show()

        if not self.firstPrediction: return

        # Draw KNN boundary

        # The boundary is just a dense grid with classified test points
        grid_axis1 = np.linspace(self.range[0], self.range[1], (self.range[1] - self.range[0]) * 10)
        grid_axis2 = np.linspace(self.range[0], self.range[1], (self.range[1] - self.range[0]) * 10)
        grid_x, grid_y = np.meshgrid(grid_axis1, grid_axis2)
        grid_points = np.array([np.array(point) for point in zip(grid_x.flatten(), grid_y.flatten())])
        labels = np.array([])
        
        training_x = np.hstack((self.blue_x, self.red_x))
        training_y = np.hstack((self.blue_y, self.red_y))
        training_data = np.stack((training_x, training_y), axis=1)
        labels = np.hstack((np.array([0] * len(self.blue_x)), np.array([1] * len(self.red_x))))
        predictions, indices = knn(training_data, labels, grid_points, self.k, 2)

        predictions[predictions == 0] = -1
        predictions = predictions.reshape(grid_y.shape)

        self.ax.contourf(grid_x, grid_y, predictions, levels=[-1, 0, 1], colors=("cyan", "salmon"), zorder=0)
        self.show()

        self.firstPrediction = False