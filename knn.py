import numpy as np
import matplotlib.pyplot as plt

def knn(training_points, labels, testing_point, k):
    n = len(training_points)
    distances = np.zeros(n)
    for i, training_point in enumerate(training_points):
        distances[i] = np.linalg.norm(training_point - testing_point)
    
    picked = np.zeros(n)
    closestDistances = np.array([np.max(distances)] * k)
    closestIndices = np.array([-1] * k)
    for i in range(0, k):
        for j in range(0, n):
            if(picked[j] == 0 and closestDistances[i] >= distances[j]):
                closestDistances[i] = distances[j]
                closestIndices[i] = j
        picked[closestIndices[i]] = 1
    ans = 0
    for i in range(0, k):
        ans += labels[closestIndices[i]]
    
    if(ans > 0): ans = 1
    elif(ans < 0): ans = -1
    elif(ans == 0): ans = np.random.choice([-1, 1], 1)

    return (ans, closestIndices)