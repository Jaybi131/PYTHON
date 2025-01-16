import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

np.random.seed(42)         
beispiel = 'iris'          
if beispiel == 'iris':
    dataset = np.loadtxt('iris.csv', delimiter=",")
    X = dataset[:,0:4]
    Y = dataset[:,4]
    names = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']   
else:
    X = np.loadtxt("BostonFeature.csv", delimiter=",")
    Y = np.loadtxt("BostonTarget.csv", delimiter=",")
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 
             'PTRATIO', 'B', 'LSTAT']

NoS = X.shape[0] 
zufall = np.random.randint(100, size=(NoS, 1)) / 10
X = np.hstack((X, zufall))
names = np.hstack((names, 'rand'))
NoF = X.shape[1] 

if beispiel == 'iris': 
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
else:                  
    rf = RandomForestRegressor(n_estimators=100, random_state=21)
rf.fit(X, Y) 

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(NoF):
    print('%d . feature %d %s;  %f.4' % (f+1, indices[f], names[indices[f]], 
                                         importances[indices[f]]))

inEveryTree = []
for tree in rf.estimators_ :
    inEveryTree.append(tree.feature_importances_)
inEveryTree = np.array(inEveryTree)
VarImportance = np.std(inEveryTree, axis=0)

plt.figure()
plt.title("Feature importances")
ind = np.arange(NoF)
plt.bar(ind, importances[indices], color="r",
        yerr=VarImportance[indices], align="center")
plt.xticks(ind, names[indices])
plt.show()

