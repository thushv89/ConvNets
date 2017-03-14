from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
import numpy as np

X1 = np.asarray([[0,1,0],[1,0,1],[0,0,1],[1,1,0],[0,0,0]])
Y1 = np.asarray([[1],[0],[0],[1],[0]])

regressor = MLPRegressor(activation='tanh', batch_size=5,
                                      hidden_layer_sizes=(64, 32), learning_rate='constant',
                                      learning_rate_init=0.001, max_iter=20,
                                      random_state=1, shuffle=True,
                                      solver='sgd', momentum=0.95)

target = regressor.partial_fit(X1,Y1)
target = clone(target)

print(regressor.predict(np.asarray([[0,1,0],[1,0,0]])))
print(target.predict(np.asarray([[0,1,0],[1,0,0]])))

X2 = np.asarray([[0,1,0],[1,0,1],[0,0,1],[1,1,0],[0,0,0]])
Y2 = np.asarray([[0],[1],[0],[1],[0]])

regressor.partial_fit(X2,Y2)

print(regressor.predict(np.asarray([[0,1,0],[1,0,0]])))

print(target.predict(np.asarray([[0,1,0],[1,0,0]])))