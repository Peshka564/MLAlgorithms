import numpy as np

class Perceptron:

    def fit(self, training_data, training_labels):
        if len(training_data) == 0: raise Exception("No training samples")
        
        n = len(training_data)
        d = len(training_data[0])

        # "Absorb" bias and init parameters
        biases = np.full(shape=(n, 1), fill_value=1)
        training_data = np.concatenate((training_data, biases), axis=1)
        w = np.zeros(shape=(d + 1))
        
        # Training loop
        while True:
            changed = False
            for i in range(0, len(training_data)):
                if training_labels[i] * np.dot(training_data[i], w) <= 0:
                    w += training_labels[i] * training_data[i]
                    changed = True
            if not changed: break
        
        self.w = w

    def get_parameters(self):
        return self.w
    
    def predict(self, testing_points):
        predictions = np.array([np.dot(self.w, testing_point) for testing_point in testing_points])
        predictions[predictions < 0] = -1
        predictions[predictions >= 0] = 1
        return predictions