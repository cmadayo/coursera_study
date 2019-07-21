import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from modules import plot_data, predict, cost_function, one_vs_all, predict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# training data
training_datafile = './training_data/ex3data1'
weight_datafile = './training_data/ex3weights'


input_layer_size = 400    # 20x20 Input Images of Digits
hidden_layer_size = 25    # 25 hidden units
num_labels = 10           # from 1 to 10 (mapped 10 to label 0)

# initalize array
data = np.array
X = np.array
y = np.array

# load data from mat file
data = loadmat(training_datafile)
X = data['X']                               # 5000 * 400
y = np.reshape(data['y'],(-1,))             # 5000 * 1

m = X.shape[0]

# select random training set
random_array = np.random.permutation(m)
selected_array = X[random_array[0:100], :]       # 100 * 400

# display data from mat
#plot_data.mat_data_plot(selected_array, 20, 20)  # 2nd 3rd param -> 20x20 Input Images of Digits

data = loadmat(weight_datafile)
Theta1 = data['Theta1']
Theta2 = data['Theta2']


# ---------------- hand make function start ----------------
result_array = predict.execute_for_neural_network(Theta1, Theta2, X)
acc = np.sum((result_array == y).astype(np.int)) * 100 / m
print(acc)
# ---------------- hand make function end   ----------------

mlpc = MLPClassifier(hidden_layer_sizes=(26, ), solver="adam", random_state=9999, max_iter=10000)
mlpc.fit(X, y)
acc = mlpc.score(X, y)*100
print(acc)
