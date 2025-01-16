import numpy as np
from binaryTree import tree 

class bDecisionTree:
    def _calGiniImpurity(self,y):
        unique, counts = np.unique(y, return_counts=True)
        N = counts/len(y)
        G = 1 - np.sum(N**2)
        return G

    def _bestSplit(self,X,y,feature):
        G = 1
        bestSplit = np.inf
        XSort = np.unique(X[:,feature].round(self.xDecimals))
        XDiff = (XSort[1:] + XSort[:-1]) / 2
        for i in range(XDiff.shape[0]):
            index = X[:,feature] < XDiff[i]
            G1 = self._calGiniImpurity(y[index])
            G2 = self._calGiniImpurity(y[~index])
            GSplit = np.mean(index)*G1 + np.mean(~index)*G2
            if G > GSplit:
                G = GSplit
                bestSplit = XDiff[i]
        return bestSplit, G

    def _chooseFeature(self,X,y):
        G         = np.zeros(X.shape[1])
        bestSplit = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            ( bestSplit[i] , G[i] ) = self._bestSplit(X,y,i)
        smallest = np.argmin(G)
        return G[smallest], bestSplit[smallest], smallest

    def _ComputeValue(self,y):
        unique, counts = np.unique(y, return_counts=True)
        i = np.argmax(counts)
        return(unique[i])

    def __init__(self,threshold = 0.1, xDecimals = 8, minLeafNodeSize=3):
        self.bTree = None
        self.threshold = threshold
        self.xDecimals = xDecimals
        self.minLeafNodeSize = minLeafNodeSize

    def _GenTree(self,X,y,parentNode,branch):
        commonValue = self._ComputeValue(y)
        initG = self._calGiniImpurity(y)
        if  initG < self.threshold or X.shape[0] <= self.minLeafNodeSize:
            self.bTree.addNode(parentNode,branch,commonValue)
            return
            
        (G, bestSplit ,chooseA) = self._chooseFeature(X,y)
        if G > 0.98*initG:
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
        
if __name__ == '__main__':        
    dataset = np.loadtxt("iris.csv", delimiter=",")

    np.random.seed(42)
    MainSet = np.arange(0,dataset.shape[0])
    Trainingsset = np.random.choice(dataset.shape[0], 120, replace=False)
    Testset = np.delete(MainSet,Trainingsset)
    XTrain = dataset[Trainingsset,0:4]
    yTrain = dataset[Trainingsset,4]
    XTest = dataset[Testset,0:4]
    yTest = dataset[Testset,4]
    
    myTree = bDecisionTree(minLeafNodeSize=5)
    myTree.fit(XTrain,yTrain)
    
    yPredict = myTree.predict(XTest)
    print(yPredict - yTest)
