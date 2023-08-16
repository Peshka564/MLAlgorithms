import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
import numpy as np
from abc import ABC, abstractmethod

class ClassificationPlot(ABC):

    def __init__(self, range):
        self.fig, self.ax = plt.subplots()
        self.mode = 0
        self.color = "blue"
        self.title = "Click to draw blue points. Press Enter to draw red points."

        self.blue_x = np.array([])
        self.blue_y = np.array([])
        self.red_x = np.array([])
        self.red_y = np.array([])

        self.range = range
        self.ax.set_xlim([self.range[0], self.range[1]])
        self.ax.set_ylim([self.range[0], self.range[1]])
        self.ax.legend(handles=[Line2D([0], [0], marker='o', color="blue", label="-1" markerfacecolor="blue", markersize=10), 
                                Line2D([0], [0], marker='o', color="red", label="1", markerfacecolor="red", markersize=10)], loc=1)

        self.fig.canvas.mpl_connect("key_press_event", self.onPress)
        self.fig.canvas.mpl_connect("button_press_event", self.onClick)

    def show(self):
        self.ax.set_title(self.title)
        plt.show()

    def onClick(self, event):
        if event.button == MouseButton.LEFT:
            self.ax.set_title(self.title)
            if self.mode == 0:
                self.blue_x = np.append(self.blue_x, event.xdata)
                self.blue_y = np.append(self.blue_y, event.ydata)
                self.ax.scatter(self.blue_x, self.blue_y, color=self.color, edgecolors='black')
            elif self.mode == 1:
                self.red_x = np.append(self.red_x, event.xdata)
                self.red_y = np.append(self.red_y, event.ydata)
                self.ax.scatter(self.red_x, self.red_y, color=self.color, edgecolors='black')
            elif self.mode == 2:
                self.additional_actions((event.xdata, event.ydata))
            self.show()
    
    def onPress(self, event):
        if event.key == "enter":
            if self.mode == 0:
                self.title = "Click to draw red points. Press Enter to classify."
                self.color = "red"
                self.mode = 1
                self.ax.set_title(self.title)
                self.show()
            elif self.mode == 1:
                self.mode = 2
                self.updateAfterPlot()
                self.show()
                            
    @abstractmethod
    def updateAfterPlot(self):
        ...

    @abstractmethod
    def classify(self, testing_point):
        ...

    def additional_actions(self, testing_point):
        pass