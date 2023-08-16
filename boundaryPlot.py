from classificationPlot import ClassificationPlot
import numpy as np

class BoundaryPlot(ClassificationPlot):
    def __init__(self, range, linear_classifier):
        super().__init__(range)
        self.linear_classifier = linear_classifier

    def updateAfterPlot(self):
        # Maybe easier for non-linear classifiers
        # boundaryFunction = np.vectorize(self.classify())
        # boundaryY = boundaryFunction(boundaryX)
        # w, b = self.classify()
        #boundaryX = np.linspace(self.range[0], self.range[1], (self.range[1] - self.range[0]) * 100)
        
        x = np.linspace(self.range[0], self.range[1], (self.range[1] - self.range[0]) * 10)
        y = np.linspace(self.range[0], self.range[1], (self.range[1] - self.range[0]) * 10)
        # params = [w, b]
        params = self.classify()
        x, y = np.meshgrid(x, y)
        grid_points = np.array([[point_x, point_y, 1] for point_x, point_y in zip(x.flatten(), y.flatten())])

        p = np.array([1 if np.dot(params, point) >= 0 else -1 for point in grid_points])
        p = p.reshape(x.shape)

        self.ax.contourf(x, y, p, levels=[-1, 0, 1], colors=("cyan", "salmon"), zorder=0)


    def classify(self):
        # Prepare data
        blues = np.array([(i, j) for i, j in zip(self.blue_x, self.blue_y)])
        reds = np.array([(i, j) for i, j in zip(self.red_x, self.red_y)])
        training_points = np.concatenate((blues, reds), axis=0)
        training_labels = np.concatenate((np.full(shape=(len(blues), 1), fill_value=-1), np.full(shape=(len(reds), 1), fill_value=1)), axis=0)
        
        self.linear_classifier.fit(training_points, training_labels)
        return self.linear_classifier.get_parameters()
        