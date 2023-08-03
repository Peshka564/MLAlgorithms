from knnPlot import KNN_Plotter
from knn import knn_eval
from scipy.io import loadmat
import numpy as np

def __main__():
    # knnp = KNN_Plotter(k=4)
    # knnp.show()

    # Cornell CS4780 course data
    data = loadmat('./MlAlgorithms/faces.mat')
    training_data = np.array(data['xTr'].T)
    training_labels = np.round(data['yTr']).T.flatten() - 1
    testing_data = np.array(data['xTe'].T)
    testing_labels = np.round(data['yTe']).T.flatten() - 1

    knn_eval(training_data, training_labels, testing_data, testing_labels, 3, 40)

__main__()