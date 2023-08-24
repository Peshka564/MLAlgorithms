import numpy as np
from abc import ABC, abstractmethod

class NaiveBayes(ABC):

    def fit(self, training_data, num_labels):

        # training_data: [(features, label)]
        if not len(training_data): raise Exception("No training data")
        if num_labels < 2: raise Exception("Invalid number of labels")

        self.num_features = training_data.shape[1] - 1
        self.num_labels = num_labels
        self.training_data = training_data

        # Assuming the labels are in the range [0, num_labels)
        # label_prob: log(P(Y))
        self.log_label_prob = self.compute_label_probabilites()

        self.compute_needed_params()
                
    @abstractmethod
    def predict(self, training_example):
        ...
        
    @abstractmethod
    def compute_needed_params(self):
        ...

    def compute_label_probabilites(self):
        # P(Y = yi) = label_freqs[i] / len(label_freqs) with +1 smoothing
        label_freqs = np.zeros(shape=self.num_labels)
        for i in range(0, len(self.training_data)):
            label_freqs[int(self.training_data[i][-1])] += 1
        return np.log(label_freqs + 1 / (len(self.training_data) + self.num_labels))
    
class MultinomialNB(NaiveBayes):

    def compute_needed_params(self):

        # feature_prob: P(X | Y)
        feature_probs = np.zeros(shape=(self.num_labels, self.num_features))

        for label_index in range(self.num_labels):
            # Compute the probability of each feature in the dataset given the label
            for feature_index in range(self.num_features):
                for training_example_index in range(len(self.training_data)):
                    if self.training_data[training_example_index][-1] == label_index:
                        feature_probs[label_index][feature_index] += self.training_data[training_example_index][feature_index] + 1

        num_words_per_label = (np.array([np.sum(feature_probs[label_index]) for label_index in range(self.num_labels)]) + self.num_features).reshape(-1, 1)
        
        # log_feature_prob: log(P(X | Y))
        self.log_feature_prob = np.log(feature_probs / num_words_per_label)

    def predict(self, training_example):
        # y = argmax ln(P(y)) + sum i from 0 to num_features - xi * ln(P(xi | y))
        probs = np.copy(self.log_label_prob)
        for label in range(self.num_labels):
            for feature_index in range(self.num_features):
                probs[label] += training_example[feature_index] * self.log_feature_prob[label][feature_index]
        return np.argmax(probs)

class GaussianNB(NaiveBayes):

    def compute_needed_params(self):
        self.means = np.zeros(shape=(self.num_labels, self.num_features))
        self.variances = np.zeros(shape=(self.num_labels, self.num_features))
        
        # Assuming every class label has a representative
        label_counts = np.bincount(self.training_data[:, -1].astype(int), minlength=self.num_labels).reshape(-1, 1)

        # Get the means
        for training_example in self.training_data:
           self.means[int(training_example[-1])] += training_example[:-1]

        self.means /= label_counts
        
        # Get the variances
        for training_example in self.training_data:
            self.variances[int(training_example[-1])] += (training_example[:-1] - self.means[int(training_example[-1])])**2

        self.variances /= label_counts

        # Adding a little gaussian noise to avoid the variance being 0
        # variances += np.random.normal(0, 1, variances.shape)

    def predict(self, training_example):
        # y = argmax ln(P(y)) + sum i from 0 to num_features - (1 / sqrt(2 * PI * variances[i])) -  (xi - means[i]) ** 2 / 4 * variances[i] ** 2
        probs = np.copy(self.log_label_prob)
        for label in range(self.num_labels):
            for feature_index in range(self.num_features):
                probs[label] +=  (1 / np.sqrt(2 * np.pi * self.variances[label][feature_index])) - (training_example[feature_index] - self.means[label][feature_index]) ** 2 / (2 * self.variances[label][feature_index])
        return np.argmax(probs)
    
# Note: When the variances are shared across labels, i.e. only one variance per feature
# the gaussian nb classifier is a linear classifier
# Otherwise GaussianNB is a quadratic classifier

class LinearGaussianNB(NaiveBayes):

    def compute_needed_params(self):
        self.means = np.zeros(shape=(self.num_labels, self.num_features))
        
        # Assuming every class label has a representative
        label_counts = np.bincount(self.training_data[:, -1].astype(int), minlength=self.num_labels).reshape(-1, 1)

        # Get the means
        for training_example in self.training_data:
           self.means[int(training_example[-1])] += training_example[:-1]

        self.means /= label_counts

        self.overall_mean = np.zeros(shape=(self.num_features))
        for training_example in self.training_data:
            self.overall_mean += training_example[:-1]
        self.overall_mean /= len(self.training_data)

        self.variances = np.zeros(shape=(self.num_features))
        for training_example in self.training_data:
            self.variances += (training_example[:-1] - self.overall_mean) ** 2
        self.variances /= len(self.training_data)


    def predict(self, training_example):
        # y = argmax ln(P(y)) + sum i from 0 to num_features - (1 / sqrt(2 * PI * variances[i])) -  (xi - means[i]) ** 2 / 4 * variances[i] ** 2
        probs = np.copy(self.log_label_prob)
        for label in range(self.num_labels):
            for feature_index in range(self.num_features):
                probs[label] +=  (1 / np.sqrt(2 * np.pi * self.variances[feature_index])) - (training_example[feature_index] - self.means[label][feature_index]) ** 2 / (2 * self.variances[feature_index])
        return np.argmax(probs)
