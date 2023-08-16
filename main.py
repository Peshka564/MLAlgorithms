from knnPlot import KNN_Plotter
from knn import knn_eval
from scipy.io import loadmat
import numpy as np
from sklearn import neighbors
from boundaryPlot import BoundaryPlot
from perceptron import Perceptron
import matplotlib.pyplot as plt
def __main__():

    # bplot = BoundaryPlot(range=(0, 10), linear_classifier=Perceptron())
    # bplot.show()
    # knnp = KNN_Plotter(k=4, range=(0, 10))
    # knnp.show()

    # Cornell CS4780 course data
    data = loadmat('./MlAlgorithms/faces.mat')
    training_data = np.array(data['xTr'].T)
    training_labels = np.round(data['yTr']).T.flatten() - 1
    testing_data = np.array(data['xTe'].T)
    testing_labels = np.round(data['yTe']).T.flatten() - 1

    k = 4
    numClasses = 40
    knn_eval(training_data, training_labels, testing_data, testing_labels, k, numClasses)

    skl_classifier = neighbors.KNeighborsClassifier(n_neighbors=k, weights="uniform")
    skl_classifier.fit(training_data, training_labels)

    skl_predictions = skl_classifier.predict(testing_data)

    correct = 0
    for i in range(0, len(testing_labels)):
        correct += skl_predictions[i] == testing_labels[i]

    print(f"SKL KNN Accuracy: {100 * correct / len(skl_predictions)}%, ({correct}/{len(skl_predictions)})")

__main__()