from classificationPlot import ClassificationPlot
import numpy as np

class BoundaryPlot(ClassificationPlot):
    def __init__(self, range):#, classifier, additional_params):
        super().__init__(range)
        #self.classifier = classifier
        #self.additional_params = additional_params

    def updateAfterPlot(self):
        # Maybe easier for non-linear classifiers
        # boundaryFunction = np.vectorize(self.classify())
        # boundaryY = boundaryFunction(boundaryX)
        #w, b = self.classify()
        #boundaryX = np.linspace(self.range[0], self.range[1], (self.range[1] - self.range[0]) * 100)
        x = np.linspace(self.range[0], self.range[1], (self.range[1] - self.range[0]) * 100)
        w = 5
        b = 5
        y = w * x + b
        self.ax.plot(x, y, c="black")
        self.ax.fill_between(x, self.range[0], y, facecolor=f"{'red' if b > 0 else 'blue'}", zorder=0)
        self.ax.fill_between(x, y, self.range[1], facecolor=f"{'blue' if b > 0 else 'red'}", zorder=0)

    def classify(self):
        pass
        #self.classifier.fit(x, y)