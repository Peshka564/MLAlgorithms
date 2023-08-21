from knnPlot import KNN_Plotter
from perceptronPlot import PerceptronPlot
from knn import knn_eval
from perceptron import Perceptron

import numpy as np
import matplotlib.pyplot as plt

from knnComparison import classifyFaces

def __main__():

    bplot = PerceptronPlot(range=(0, 10))
    bplot.show()
    #knnp = KNN_Plotter(k=4, range=(0, 10))
    #knnp.show()

    #classifyFaces()

__main__()