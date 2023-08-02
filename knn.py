import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker

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

def addPoint(coords, type):
    global x, y, z
    x = np.append(x, coords[0])
    y = np.append(y, coords[1])
    z = np.append(z, int(type))

#####################
# plt.ion()
# fig, ax = plt.subplots(constrained_layout=True)

# x = np.array([])
# y = np.array([])
# z = np.array([])

# ax.scatter(x, y)
# ax.legend(loc=1)

# klicker = clicker(ax, ["-1", "1"], markers=["x", "o"], linestyle="-")
# klicker.on_point_added(addPoint)

# input("Something")
# klicker = clicker(ax, ["0"], markers=["*"])
# plt.show()
