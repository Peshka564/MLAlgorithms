import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from algorithms.naiveBayes import NaiveBayes
from sklearn.naive_bayes import MultinomialNB

def classifyGenderByName():
    def hashfeatures(baby, B):
        v = np.zeros(B)
        for letter in baby:
            v[hash(letter) % B] = 1
        return v

    def name2features(filename, B=128, LoadFile=True):
        # read in baby names
        if LoadFile:
            with open(filename, 'r') as f:
                babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
        else:
            babynames = filename.split('\n')
        n = len(babynames)
        X = np.zeros((n, B))
        for i in range(n):
            X[i,:] = hashfeatures(babynames[i], B)
        return X

    def genTrainFeatures(dimension=128):

        # Load in the data
        Xgirls = name2features("./MlAlgorithms/data/girls.train", B=dimension)
        Xboys = name2features("./MlAlgorithms/data/boys.train", B=dimension)
        X = np.concatenate([Xgirls, Xboys])
        
        # Generate Labels
        Y = np.concatenate([np.ones(len(Xgirls)), np.zeros(len(Xboys))])
        
        # shuffle data into random order
        ii = np.random.permutation([i for i in range(len(Y))])
        
        return X[ii, :], Y[ii]

    X,Y = genTrainFeatures(128)
    Y = Y.reshape(-1, 1)

    nb = NaiveBayes([1] * 128)
    nbx = MultinomialNB()

    nb.fit(np.concatenate((X, Y), axis=1), 2)
    nbx.fit(X, Y)

    testing_names = np.array(["Johnny", "Trump", "Pearl", "Adriana", "Melina", "DJ", "Caitlyn"])
    hashed_names = np.empty(shape=(len(testing_names), 128))

    for i in range(len(testing_names)):
        hashed_names[i] = hashfeatures(testing_names[i], 128)

    for i in range(len(testing_names)):
        print(f"Original: {testing_names[i]} is a {('boy' if nb.predict(hashed_names[i]) == 0 else 'girl')}")
        print(f"Sklearn: {testing_names[i]} is a {('boy' if nbx.predict(np.array([hashed_names[i]])) == 0 else 'girl')}")
        print()
        #print(hashed_names[i])

