import numpy as np
import matplotlib.pyplot as plt     

# In case of tie -> choose randomly
def random_choice(labels, closestIndices, numberOfClasses, k):
    labelFreq = np.zeros(numberOfClasses)
    for i in range(0, k):
        labelFreq[labels[closestIndices[i]]] += 1

    maxFreq = -1
    maxLabel = -1
    for i in range(0, numberOfClasses):
        if labelFreq[i] > maxFreq:
            maxFreq = labelFreq[i]
            maxLabel = i
        elif labelFreq[i] == maxFreq:
            maxLabel = np.random.choice([maxLabel, i], 1)[0]
    return maxLabel

# In case of tie -> decrease k until tie is broken
def decreaseK(labels, closestIndices, numberOfClasses):
    labelFreq = np.zeros(numberOfClasses)
    labelFreq[labels[closestIndices[0]]] += 1
    candidates = [labels[closestIndices[0]]]
    tied_candidates = [1]
    occurances = 1

    for i in range(1, len(closestIndices)):
        labelFreq[labels[closestIndices[i]]] += 1
        if labelFreq[labels[closestIndices[i]]] > occurances:
            occurances += 1
            candidates.append(labels[closestIndices[i]])
            tied_candidates.append(1)
        elif labelFreq[labels[closestIndices[i]]] == occurances:
            tied_candidates.append(tied_candidates[i - 1] + 1)
            candidates.append(candidates[i - 1])
        else:
            candidates.append(candidates[i - 1])
            tied_candidates.append(tied_candidates[i - 1])

    for i in range(len(closestIndices) - 1, -1, -1):
        if tied_candidates[i] == 1:
            return candidates[i]

# Additional improvements to add:
    # Learn the best metric during training
    # Feature weights
    # Data augmentation
    # Compare with scikit classifier

def knn(training_points, labels, testing_points, k, numberOfClasses, normOrder=2):

    n = len(training_points)
    m = len(testing_points)
    predicted_labels = np.empty(shape=(m))
    closest = np.empty(shape=(m, k), dtype=int)

    for testing_index, testing_point in enumerate(testing_points):
         
        # Find distances from each testing point to each training point
        distances = np.zeros(n)
        for j, training_point in enumerate(training_points):
            distances[j] = np.linalg.norm(testing_point - training_point, ord=normOrder)
    
        # Find the k smallest distances + corresponding indices
        picked = np.zeros(n)
        closestDistances = np.full(shape=(k), fill_value=np.max(distances))
        closestIndices = np.full(shape=(k), fill_value=-1)
        for currentPosition in range(0, k):
            for training_index in range(0, n):
                if(picked[training_index] == 0 and closestDistances[currentPosition] >= distances[training_index]):
                    closestDistances[currentPosition] = distances[training_index]
                    closestIndices[currentPosition] = training_index
            picked[closestIndices[currentPosition]] = 1

        # Classify points
        # Expecting labels[i] to be from 0 to numberOfLabels
        predicted_labels[testing_index] = decreaseK(labels, closestIndices, numberOfClasses)
        closest[testing_index] = closestIndices

    return (predicted_labels, closest)

def knn_eval(training_points, training_labels, testing_points, testing_labels, k, numberofClasses, normOrder=2):
    predicted_labels, indices = knn(training_points, training_labels, testing_points, k, numberofClasses, normOrder)
    
    correct = 0
    for i in range(0, len(testing_labels)):
        correct += predicted_labels[i] == testing_labels[i]

    print(f"KNN Accuracy: {100 * correct / len(testing_points)}%, ({correct}/{len(testing_points)})")