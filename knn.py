import numpy as np
import matplotlib.pyplot as plt

def knn(training_points, labels, testing_point, k):
    # Find distances from testing point to each training point
    n = len(training_points)
    distances = np.zeros(n)
    for i, training_point in enumerate(training_points):
        distances[i] = np.linalg.norm(training_point - testing_point)
    
    # Find the k smallest distances + corresponding indices
    picked = np.zeros(n)
    closestDistances = np.array([np.max(distances)] * k)
    closestIndices = np.array([-1] * k)
    for i in range(0, k):
        for j in range(0, n):
            if(picked[j] == 0 and closestDistances[i] >= distances[j]):
                closestDistances[i] = distances[j]
                closestIndices[i] = j
        picked[closestIndices[i]] = 1

    # Choose which label to classify
    # If the number of labels for each class matches,
    # Reduce k from even to odd
    label_sum = 0
    for j in range(0, k):
        label_sum += labels[closestIndices[j]]

    if label_sum == 0:
        label_sum -= labels[closestIndices[-1]]
        closestIndices = closestIndices[:-1]
    
    if(label_sum > 0): label_sum = 1
    elif(label_sum < 0): label_sum = -1
    #elif(label_sum == 0): label_sum = np.random.choice([-1, 1], 1)

    return (label_sum, closestIndices)