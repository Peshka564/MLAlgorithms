import numpy as np

class NaiveBayes():

    def __init__(self, distributions):
        self.distributions = distributions

    def fit(self, training_data, num_labels, dict_size):

        # training_data: [(features, label)]
        if not len(training_data): raise Exception("No training data")
        if num_labels < 2: raise Exception("Invalid number of labels")
        num_features = training_data.shape[1] - 1
        if num_features != len(self.distributions): raise Exception("Invalid distribution-feature mapping")

        self.num_features = num_features
        self.num_labels = num_labels

        # P(Y = yi) = label_freqs[i] / len(label_freqs)
        # Assuming the labels are in the range [0, num_labels)
        label_freqs = self.compute_label_probabilites(training_data, self.num_labels)

        # label_prob: log(P(Y))
        self.label_prob = np.log(label_freqs / len(training_data))

        # P(X | Y)
        feature_freqs = np.zeros(shape=(self.num_labels, self.num_features))

        for label_index in range(self.num_labels):
            # Compute the probability of each feature in the dataset given the label
            for feature_index in range(self.num_features):
                for training_example_index in range(len(training_data)):
                    if training_data[training_example_index][-1] == label_index:
                        feature_freqs[label_index][feature_index] += training_data[training_example_index][feature_index] + 1

        num_words_per_label = np.array([np.sum(feature_freqs[label_index]) for label_index in range(num_labels)]) + dict_size
        
        # feature_prob: log(P(X | Y))
        self.feature_prob = np.empty(shape=feature_freqs.shape) 
        for label in range(num_labels):
            self.feature_prob[label] = np.log(feature_freqs[label] / num_words_per_label[label])
                
    def predict(self, training_example):
        # y = argmax ln(P(y)) + sum i from 0 to num_features - xi * ln(P(xi | y))
        probs = np.copy(self.label_prob)
        for label in self.num_labels:
            for feature_index in self.num_features:
                probs[label] += training_example[feature_index] * self.feature_prob[label][feature_index]
        return np.argmax(probs)

    def compute_label_probabilites(training_data, num_labels):
        label_freqs = np.zeros(shape=num_labels)
        for i in range(0, len(training_data)):
            label_freqs[training_data[i][-1]] += 1
        return label_freqs