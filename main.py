import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import plotData

# training data
training_datafile = './ex2data1.txt'

# read csvdata
csvdata = pd.read_csv(training_datafile, header=None)

# initalize array
data = np.array
X = np.array
y = np.array

data = csvdata.values
X = data[:,0:2]         # 100 * 2
y = data[:,2]           # 100 * 1

plotData.plot(X, y)
