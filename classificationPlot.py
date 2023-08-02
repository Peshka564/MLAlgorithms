from knn import knn
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
import numpy as np

class ClassificationPlot:

    def __init__(self, classifier):
        self.fig, self.ax = plt.subplots()
        self.classifier = classifier
        self.mode = -1
        self.color = "blue"
        self.title = "Click to draw blue points. Press Enter to draw red points."
        self.firstPrediction = True

        self.blue_x = np.array([])
        self.blue_y = np.array([])
        self.red_x = np.array([])
        self.red_y = np.array([])

        self.ax.set_xlim([0, 10])
        self.ax.set_ylim([0, 10])
        self.ax.legend(handles=[Line2D([0], [0], marker='o', color="blue", label="-1", markerfacecolor="blue", markersize=10), 
                                Line2D([0], [0], marker='o', color="red", label="1", markerfacecolor="red", markersize=10)], loc=1)

        self.fig.canvas.mpl_connect("key_press_event", self.onPress)
        self.fig.canvas.mpl_connect("button_press_event", self.onClick)

    def show(self):
        self.ax.set_title(self.title)
        plt.show()

    def onClick(self, event):
        if event.button == MouseButton.LEFT:
            self.ax.set_title(self.title)
            if self.mode == -1:
                self.blue_x = np.append(self.blue_x, event.xdata)
                self.blue_y = np.append(self.blue_y, event.ydata)
                self.ax.scatter(self.blue_x, self.blue_y, color=self.color)
            elif self.mode == 1:
                self.red_x = np.append(self.red_x, event.xdata)
                self.red_y = np.append(self.red_y, event.ydata)
                self.ax.scatter(self.red_x, self.red_y, color=self.color)
            elif self.mode == 0:
                self.classify((event.xdata, event.ydata))
                self.firstPrediction = False
            self.show()
    
    def onPress(self, event):
        if event.key == "enter":
            if self.mode == -1:
                self.title = "Click to draw red points. Press Enter to classify."
                self.color = "red"
                self.mode = 1
                self.ax.set_title(self.title)
                self.show()
            elif self.mode == 1:
                self.title = "Click to plot test point"
                self.color = "black"
                self.mode = 0
                self.ax.set_title(self.title)
                self.show()

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
        labels = np.hstack((np.array([-1] * len(self.blue_x)), np.array([1] * len(self.red_x))))
        label, indices = self.classifier(training_data, labels, testing_point, 5)

        # Drawing prediction
        for index in indices:
            # connect points to visualize nearest neighbours
            self.temporaries.append(self.ax.plot([training_data[index][0], testing_point[0]], [training_data[index][1], testing_point[1]], color="green")[0])
        self.temporaries.append(self.ax.scatter(testing_point[0], testing_point[1], color=self.color))
        self.title = f"The labels is: {'Blue' if label == -1 else 'Red'}"
        self.ax.set_title(self.title)
        self.show()

cp = ClassificationPlot(knn)
cp.show()