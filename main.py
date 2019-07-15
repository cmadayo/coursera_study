import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from modules import plot_data, predict, cost_function

# training data
training_datafile = './training_data/ex2data1.txt'

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

# size analysis
m = X.shape[0]
n = X.shape[1]

# Add ones to X
X = np.insert(X, 0, 1, axis=1)

# initial value of theta
initial_theta = np.zeros((n + 1))

# wrapper for fmin
costFunction_wrapper = lambda theta, X, y: cost_function.get_cost_grad(theta, X, y)[0]
# minimize costFunction's cost
result = fmin(costFunction_wrapper, initial_theta, args=(X, y,),full_output=True, disp=False)
# minimize costFunction's cost result
theta, cost = result[0], result[1]

# plot boundary
plot_data.scatter_plot_border(theta, X, y)

# calculate accuracy
p = predict.execute(theta, X, 0.5)
print(np.mean(p == y)*100)
