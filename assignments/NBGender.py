import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from algorithms.naiveBayes import MultinomialNB
#from sklearn.naive_bayes import MultinomialNB

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
        Xgirls = name2features("./MlAlgorithms/data/girls.csv", B=dimension)
        Xboys = name2features("./MlAlgorithms/data/boys.csv", B=dimension)
        X = np.concatenate([Xgirls[:500], Xboys[:500]])
        
        # Generate Labels
        Y = np.concatenate([np.ones(len(Xgirls[:500])), np.zeros(len(Xboys[:500]))])
        
        # shuffle data into random order
        ii = np.random.permutation([i for i in range(len(Y))])
        
        return X[ii, :], Y[ii]

    X,Y = genTrainFeatures(128)
    Y = Y.reshape(-1, 1)

    nb = MultinomialNB()
    #nbx = MultinomialNB()

    nb.fit(np.concatenate((X, Y), axis=1), 2)
    #nbx.fit(X, Y)

    testing_boys = (name2features("./MlAlgorithms/data/boys.csv", B=128))[500:]
    testing_girls = (name2features("./MlAlgorithms/data/girls.csv", B=128))[500:]

    testing_names = np.concatenate((testing_boys, testing_girls), axis=0)
    hashed_names = np.empty(shape=(len(testing_names), 128))

    for i in range(len(testing_names)):
        hashed_names[i] = hashfeatures(testing_names[i], 128)

    testing_labels = np.concatenate([np.ones(len(testing_girls)), np.zeros(len(testing_boys))])

    ii = np.random.permutation([i for i in range(len(testing_labels))])

    hashed_names = hashed_names[ii, :]
    testing_labels = testing_labels[ii]
    testing_names = testing_names[ii, :]

    error = 0.0
    for i in range(len(testing_names)):
        if nb.predict(hashed_names[i]) == testing_labels[i]:
            error += 1
        #print(f"Original: {testing_names[i]} is a {('boy' if nb.predict(hashed_names[i]) == 0 else 'girl')}")
        #print(f"Sklearn: {testing_names[i]} is a {('boy' if nbx.predict(np.array([hashed_names[i]])) == 0 else 'girl')}")
        #print()
        #print(hashed_names[i])

    print(f"Error: {error * 100 / len(testing_names)}%")
