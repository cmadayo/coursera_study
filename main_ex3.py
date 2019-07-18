import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from modules import plot_data, predict, cost_function
from sklearn.linear_model import LogisticRegression


# training data
training_datafile = './training_data/ex3data1'


input_layer_size = 400    # 20x20 Input Images of Digits
num_labels = 10           # from 1 to 10 (mapped 10 to label 0)

# initalize array
data = np.array
X = np.array
y = np.array

# load data from mat file
data = loadmat(training_datafile)
X = data['X']             # 5000 * 400
y = data['y']             # 5000 * 1

m = X.shape[0]

# select random training set
random_array = np.random.permutation(m)
selected_array = X[random_array[0:100], :]       # 100 * 400

# display data from mat
#plot_data.mat_data_plot(selected_array, 20, 20)  # 2nd 3rd param -> 20x20 Input Images of Digits

theta_temp = np.array([[-2], [-1], [1], [2]])
X_temp = np.c_[np.ones((5,1)), np.reshape(np.arange(1, 16), (5, 3), 'F')/10]
y_temp = (np.array([[1], [0], [1], [0], [1]]) >= 0.5).astype(np.int)          # boolean to binary
lambda_temp = 3

# my function
(J, grad) = cost_function.get_reg_cost_grad(theta_temp, X_temp, y_temp, lambda_temp)
print('My function Result: cost = {}, gradients = {}'.format(J, grad))

# library costFunction
result = LogisticRegression(multi_class="ovr", solver="newton-cg")
print(result)
#print('My function Result: cost = {}, gradients = {}'.format(J, grad))





#X = np.arraydata

# # initalize array
# data = np.array
# X = np.array
# y = np.array
#
# data = csvdata.values
# X = data[:,0:2]         # 100 * 2
# y = data[:,2]           # 100 * 1
#
# # plot data
# plot_data.scatter_plot(X, y)
#
# # size analysis
# m = X.shape[0]
# n = X.shape[1]
#
# # Add ones to X
# X = np.insert(X, 0, 1, axis=1)
#
# # initial value of theta
# initial_theta = np.zeros((n + 1))
#
# # wrapper for fmin
# cost_function_wrapper = lambda theta, X, y: cost_function.get_cost_grad(theta, X, y)[0]
# # minimize costFunction's cost
# result = fmin(cost_function_wrapper, initial_theta, args=(X, y,),full_output=True, disp=False)
# # minimize costFunction's cost result
# theta, cost = result[0], result[1]
#
# # plot boundary
# plot_data.scatter_plot_border(theta, X, y)
#
# # calculate accuracy
# p = predict.execute(theta, X, 0.5)
# print(np.mean(p == y)*100)
