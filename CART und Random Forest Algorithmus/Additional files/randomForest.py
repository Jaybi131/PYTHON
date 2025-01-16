import numpy as np
from CARTRegressionTreeRF import bRegressionTree

class randomForestRegression:
    def __init__(self,noOfTrees=10, max_features=None, threshold = 10**-8, xDecimals = 8, minLeafNodeSize=3, perc=1):
        self.perc = perc
        self.bTree = []
        self.noOfTrees = noOfTrees
        for i in range(noOfTrees):
            tempTree = bRegressionTree(max_features=max_features, threshold=threshold, xDecimals=xDecimals, minLeafNodeSize=minLeafNodeSize)
            self.bTree.append(tempTree)

    def fit(self,X,y):
        for i in range(self.noOfTrees):
            bootstrapSample = np.random.randint(X.shape[0],size=int(self.perc*X.shape[0]))
            bootstrapX = X[bootstrapSample,:]
            bootstrapY = y[bootstrapSample]
            self.bTree[i].fit(bootstrapX,bootstrapY)

    def predict(self,X):
        ypredict = np.zeros(X.shape[0])
        for i in range(self.noOfTrees):
            ypredict += self.bTree[i].predict(X)
        ypredict = ypredict/self.noOfTrees
        return ypredict

if __name__ == '__main__':
    f = open("hourCleanUp.csv")
    header = f.readline().rstrip('\n')
    featureNames = header.split(',')
    dataset = np.loadtxt(f, delimiter=",")
    f.close()

    X = dataset[:,0:13]
    Y = dataset[:,15]

    index = np.flatnonzero(X[:,8]==4)
    X = np.delete(X,index, axis=0)
    Y = np.delete(Y,index, axis=0)

    np.random.seed(42)
    MainSet = np.arange(0,X.shape[0])
    Trainingsset = np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)
    Testset = np.delete(MainSet,Trainingsset)
    XTrain = X[Trainingsset,:]
    yTrain = Y[Trainingsset]
    XTest = X[Testset,:]
    yTest = Y[Testset]

    myForest = randomForestRegression(noOfTrees=24,minLeafNodeSize=5,threshold=2)
    myForest.fit(XTrain,yTrain)
    yPredict = np.round(myForest.predict(XTest))
    yDiff = yPredict - yTest
    print('Mittlere Abweichung: %e ' % (np.mean(np.abs(yDiff))))
