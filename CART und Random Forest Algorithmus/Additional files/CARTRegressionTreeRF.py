# %%
import numpy as np
from binaryTree import tree

class bRegressionTree:
    def _calLRSS(self,y):
        yMean = np.mean(y)
        L2 = np.sum( (y-yMean)**2)
        return L2

    def _bestSplit(self,X,y,feature):
        RSS = np.inf
        bestSplit = np.inf
        XSort = np.unique(X[:,feature].round(self.xDecimals))
        XDiff = (XSort[1:] + XSort[:-1]) / 2
        for i in range(XDiff.shape[0]):
            index = X[:,feature] < XDiff[i]
            if not (np.all(index) or np.all(~index)):
                RSS_1 = self._calLRSS(y[index])
                RSS_2 = self._calLRSS(y[~index])
                RSSSplit = RSS_1 + RSS_2
                if RSS > RSSSplit:
                    RSS = RSSSplit
                    bestSplit = XDiff[i]
        return bestSplit, RSS

    def _ComputeValue(self,y):
        return np.mean(y)

    def _chooseFeature(self, X, y):
        G         = np.inf*np.ones(X.shape[1])
        bestSplit = np.zeros(X.shape[1])

        if self.max_features is None:
            feature = np.arange(X.shape[1])
        elif self.max_features == 'sqrt':
            feature = np.random.choice(X.shape[1],int(np.sqrt(X.shape[1])),replace=False)
        else:
            feature = np.random.choice(X.shape[1],self.max_features,replace=False)

        for i in feature:
            bestSplit[i], G[i] = self._bestSplit(X,y,i)

        smallest = np.argmin(G)
        return G[smallest], bestSplit[smallest], smallest

    def __init__(self, max_features=None, threshold = 10**-8, xDecimals = 8, minLeafNodeSize=3):
        self.max_features = max_features
        self.bTree = None
        self.threshold = threshold
        self.xDecimals = xDecimals
        self.minLeafNodeSize = minLeafNodeSize

    def _GenTree(self,X,y,parentNode,branch):
        commonValue = self._ComputeValue(y)
        initG = self._calLRSS(y)
        if  initG < self.threshold or X.shape[0] <= self.minLeafNodeSize:
            self.bTree.addNode(parentNode,branch,commonValue)
            return

        (G, bestSplit ,chooseA) = self._chooseFeature(X,y)
        if G > initG:
            self.bTree.addNode(parentNode,branch,commonValue)
            return

        if parentNode == None:
            self.bTree = tree(chooseA, bestSplit, '<')
            myNo = 0
        else:
            myNo = self.bTree.addNode(parentNode,branch,bestSplit,operator='<',varNo=chooseA)

        index = np.less(X[:,chooseA],bestSplit)
        XTrue  = X[index,:]
        yTrue  = y[index]
        XFalse = X[~index,:]
        yFalse = y[~index]

        if XTrue.shape[0] > self.minLeafNodeSize:
            self._GenTree(XTrue,yTrue,myNo,True)
        else:
            commonValue = self._ComputeValue(yTrue)
            self.bTree.addNode(myNo,True,commonValue)
        if XFalse.shape[0] > self.minLeafNodeSize:
            self._GenTree(XFalse,yFalse,myNo,False)
        else:
            commonValue = self._ComputeValue(yFalse)
            self.bTree.addNode(myNo,False,commonValue)
        return()

    def fit(self, X,y):
        self._GenTree(X,y,None,None)

    def predict(self, X):
        return(self.bTree.eval(X))

    def decision_path(self, X):
        return(self.bTree.trace(X))

    def weightedPathLength(self,X):
        return(self.bTree.weightedPathLength(X))

    def numberOfLeafs(self):
        return(self.bTree.numberOfLeafs())

# %%
if __name__ == '__main__':
    np.random.seed(42)
    numberOfSamples = 10000
    X = np.random.rand(numberOfSamples,2)
    Y = ( np.sin(2*np.pi*X[:,0]) + np.cos(np.pi*X[:,1])) * np.exp(1 -X[:,0]**2 -X[:,1]**2 )

    MainSet = np.arange(0,X.shape[0])
    Trainingsset = np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)
    Testset = np.delete(MainSet,Trainingsset)

    regressionError = np.zeros(5)
    for i in range(5):
        errorRate = 0.05*i
        errorFactor = 1 + 2*(np.random.rand(Trainingsset.shape[0]) - 0.5)*errorRate
        XTrain = X[Trainingsset,:]
        yTrain = Y[Trainingsset] * errorFactor
        XTest = X[Testset,:]
        yTest = Y[Testset]

        myTree = bRegressionTree(xDecimals=3)
        myTree.fit(XTrain,yTrain)
        yPredict = myTree.predict(XTest)
        yDiff = np.abs(yPredict - yTest)
        regressionError[i] = np.mean(yDiff)

    import matplotlib.pyplot as plt
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1,1,1)
    x = np.arange(0,0.25,0.05)
    ax.plot(x,regressionError,'o-')
    ax.set_xlabel('% Noise')
    ax.set_ylabel('Mean Absolute Error')
    print(regressionError)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(1,1,1, projection='3d')
    ax.scatter(XTest[:,0],XTest[:,1],yPredict,alpha=0.6,c =yPredict, cmap=cm.jet)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_zlabel('yPredict', rotation=90)
