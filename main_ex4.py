import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from modules import plot_data, predict, cost_function, one_vs_all
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# training data
training_datafile = './training_data/ex4data1'

input_layer_size = 400    # 20x20 Input Images of Digits
hidden_layer_size = 25    # 25 hidden units
num_labels = 10           # from 1 to 10 (mapped 10 to label 0)

# load data from mat file
data = loadmat(training_datafile)
X = data['X']
y = np.reshape(data['y'], (-1, ))
m = X.shape[0]

# Randomly select 100 data points to display
sel = np.arange(m)
np.random.shuffle(sel)
train_ind = sel[0:3000]
test_ind = sel[3000:m]

X_tr = X[train_ind]
y_tr = y[train_ind]

X_ts = X[test_ind]
y_ts = y[test_ind]


# omit display data...

alpha_val = 0.0001
mlpc = MLPClassifier(hidden_layer_sizes=(26, ), solver="adam", random_state=9999, max_iter=10000, alpha=alpha_val)
mlpc.fit(X_tr, y_tr)
acc = mlpc.score(X_ts, y_ts)*100
print('acc={}'.format(acc))
