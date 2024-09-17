import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

data = np.loadtxt('accidental-deaths-in-usa-monthly.csv', delimiter=',', skiprows=1, usecols=1)
data = data.reshape(-1,1)
assert data.shape == (72, 1), f'Das Einlesen hat nicht geklappt. Shape ist: {data.shape}'

def lagged_data(data, lags):
    max_delay = max(lags)
    lagged = [ [data[i - delay, 0] for delay in lags] for i in range(max_delay, len(data)) ]
    return np.array(lagged)
    
indices = lagged_data(np.arange(72).reshape(-1, 1), [3, 1, 0])
print(indices[[0, 1, 2, -3, -2, -1]])

# Creating test data
x = np.arange(10).reshape(-1, 1)
ones = np.ones((len(x), 1))
#y = 2 * x + 1  # Gerade
y = x**2 + 1  # Parabel
X = np.hstack((ones, x, x**2))

# Lösen der Gleichung mittels Least-squares.
w = np.linalg.lstsq(X, y, rcond=None)[0]
#die Ausgabe der Koeffizienten und ein Plot entsprechend der Folien:
print(w.squeeze())

y_pred = X @ w   # Matrix-Vektor-Multiplikation
plt.figure()
plt.plot(x, y_pred)
plt.scatter(x, y, color='r')

class Linearer_Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, include_intercept=True):
        self.include_intercept = include_intercept
        self.coefficients_ = None

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        
        if self.include_intercept:
            X = self._add_intercept_column(X)

        XTX = np.dot(X.T, X)
        XTy = np.dot(X.T, y)
        self.coefficients_ = np.linalg.solve(XTX, XTy)

        return self

    def predict(self, X):
        X = self._check_X(X)
        
        if self.include_intercept:
            X = self._add_intercept_column(X)

        y_pred = np.dot(X, self.coefficients_)
        return y_pred

    def _add_intercept_column(self, X):
        intercept_column = np.ones((len(X), 1))
        return np.hstack((intercept_column, X))

    def _check_X_y(self, X, y):
        return check_X_y(X, y)

    def _check_X(self, X):
        return check_array(X)
        
x = np.arange(10).reshape(-1, 1)
#y = 2 * x + 1     # Gerade
y = x ** 2 + 1  # Parabel
X = np.hstack((x, x**2))

# Objekt erstellen
linreg = Linearer_Regressor()  # Beispiel-Name
linreg.fit(X, y.squeeze())
y_pred = linreg.predict(X)

# Koeffizienten ausgeben
print(linreg.coefficients_)  # coef_ ist auch nur ein Beispielname

plt.figure()
plt.plot(x, y_pred)
plt.scatter(x, y, color='r')

timeseries = lagged_data(data, lags=[24, 13, 12, 1, 0])

X_train = timeseries[:-12, :-1]
y_train = timeseries[:-12, -1]
y_test = timeseries[-12:, -1]

# Den eigenen Regressor aufrufen
lin_regressor = Linearer_Regressor()

timeseries = lagged_data(data, [24, 13, 12, 1, 0])

X_train = timeseries[:-12, :-1]
y_train = timeseries[:-12, -1]
y_test = timeseries[-12:, -1]

# Gewichtung analysieren (fit Methode)
lin_regressor.fit(X_train, y_train)

# Alles in einem
X_test_all = timeseries[-12:, :-1]
y_pred_all = lin_regressor.predict(X_test_all)

# Schrittweise
data_copy = data.copy()
n = len(data)
lags = np.array([24, 13, 12, 1])

for month in range(12):
    # X_test bestimmen
    lagged_values = [data_copy[-lag] for lag in lags]
    X_test_step = np.array(lagged_values).reshape(1, -1)
    
    # predict dafür
    y_pred_step =lin_regressor.predict(X_test_step)
    
    # diese dann in data_copy speichern
    data_copy = np.vstack([data_copy, y_pred_step])

# kopieren (extract) der erstellten Schrittweisen Vorhersagen
y_pred_step_by_step = data_copy[-12:, -1]

# true data vs Schrittweise Vorhersagen
plt.figure(figsize=(10, 5))
plt.title('Schrittweise Vorhersagen')
plt.plot(y_test, color='blue', label='True Data')
plt.plot(y_pred_step_by_step, color='red', label='Step-by-step Prediction')
plt.legend()
plt.show()

# true data vs alles in einem
plt.figure(figsize=(10, 5))
plt.title('Batch Vorhersagen')
plt.plot(y_test, color='blue', label='True Data')
plt.plot(y_pred_all, color='green', label='All-at-once Prediction')
plt.legend()
plt.show()

# mean absolute errors (MAE) bestimmen
mae_all = np.mean(np.abs(y_pred_all - y_test))
mae_step = np.mean(np.abs(y_pred_step_by_step - y_test))

print("MAE für Batch Vorhersagen:", mae_all)
print("MAE für Schrittweise Vorhersagen:", mae_step)
