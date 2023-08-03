import numpy as np
import matplotlib.pyplot as plt

def knn(training_points, labels, testing_points, k):
    # Find distances from each testing point to each training point
    n = len(training_points)
    m = len(testing_points)
    distances = np.zeros(shape=(m, n))
    for i, testing_point in enumerate(testing_points):
        for j, training_point in enumerate(training_points):
            distances[i][j] = np.linalg.norm(training_point - testing_point)
    
    # Find the k smallest distances + corresponding indices
    picked = np.zeros(shape=(m, n))
    closestDistances = np.full(shape=(m, k), fill_value=np.max(distances))
    closestIndices = np.full(shape=(m, k), fill_value=-1)
    for testing_index in range(0, m):
        for currentPosition in range(0, k):
            for training_index in range(0, n):
                if(picked[testing_index][training_index] == 0 and closestDistances[testing_index][currentPosition] >= distances[testing_index][training_index]):
                    closestDistances[testing_index][currentPosition] = distances[testing_index][training_index]
                    closestIndices[testing_index][currentPosition] = training_index
            picked[testing_index][closestIndices[testing_index][currentPosition]] = 1
    print(closestDistances)
    print(closestIndices)
    # Classify points
    label_sums = np.zeros(shape=(m))
    for i in range(0, m):
        for j in range(0, k):
            label_sums[i] += labels[closestIndices[i][j]]
        if label_sums[i] == 0:
            # In case of tie -> choose random            
            label_sums[i] = np.random.choice([-1, 1], 1)

            # In case of tie -> reduce k from even to odd
            # label_sums[i] -= labels[closestIndices[i][-1]]
            # closestIndices[i] = closestIndices[i][:-1]  
        elif(label_sums[i] > 0): label_sums[i] = 1
        elif(label_sums[i] < 0): label_sums[i] = -1

    return (label_sums, closestIndices)