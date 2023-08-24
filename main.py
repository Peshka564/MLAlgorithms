from plots.knnPlot import KNN_Plotter
from plots.perceptronPlot import PerceptronPlot
from plots.naiveBayesPlot import NBPlot
from algorithms.knn import knn_eval
from algorithms.perceptron import Perceptron

import numpy as np
import matplotlib.pyplot as plt

from assignments.NBGender import classifyGenderByName
def __main__():

    #nbplot = NBPlot(range=(0, 10))
    #nbplot.show()

    classifyGenderByName()
__main__()