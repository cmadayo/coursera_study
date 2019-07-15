import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs
from modules import plot_data, predict, cost_function, map_feature
import math

# training data
training_datafile = './training_data/ex2data2.txt'

# read csvdata
csvdata = pd.read_csv(training_datafile, header=None)

# initalize array
data = np.array
X = np.array
y = np.array

data = csvdata.values
X = data[:,0:2]         # 100 * 2
y = data[:,2]           # 100 * 1

# plot data
plot_data.scatter_plot(X, y)

# make feature(Too many)
X = map_feature.get_feature(X[:, 0], X[:, 1], 6)

# size analysis
m = X.shape[0]
n = X.shape[1]

# initial value of theta
initial_theta = np.zeros((n,))

# get cost,grad
lambda_value = 1
(cost, grad) = cost_function.get_reg_cost_grad(initial_theta, X, y, lambda_value)
print(cost, grad)

test_theta = np.ones((n,))
lambda_value = 10
(cost, grad) = cost_function.get_reg_cost_grad(test_theta, X, y, lambda_value)
#print(cost, grad)


# initial value of theta
initial_theta = np.zeros((n,))
lambda_value = 0

# wrapper for fmin
cost_function_reg_wrapper = lambda theta, X, y, lambda_value: cost_function.get_reg_cost_grad(theta, X, y, lambda_value)[0]

# minimize costFunction's cost
result = fmin_bfgs(cost_function_reg_wrapper, initial_theta, args=(X, y, lambda_value,), full_output=True, disp=True)

# minimize costFunction's cost result
theta, cost = result[0], result[1]

#print(theta, cost)

# plot boundary
plot_data.plot_decision_boundary(theta, X, y)

# calculate accuracy
p = predict.execute(theta, X, 0.5)
print(np.mean(p == y)*100)
