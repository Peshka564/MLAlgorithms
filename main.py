from knnPlot import KNN_Plotter
from knn import knn_eval
import numpy as np
from boundaryPlot import BoundaryPlot
from perceptron import Perceptron
import matplotlib.pyplot as plt
from knnComparison import classifyFaces

def __main__():

    # bplot = BoundaryPlot(range=(0, 10), linear_classifier=Perceptron())
    # bplot.show()
    knnp = KNN_Plotter(k=4, range=(0, 10))
    knnp.show()

    #classifyFaces()

__main__()