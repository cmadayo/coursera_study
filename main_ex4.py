import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from modules import plot_data, predict, cost_function, one_vs_all
from sklearn.linear_model import LogisticRegression

# training data
training_datafile = './training_data/ex4data1'

input_layer_size = 400    # 20x20 Input Images of Digits
hidden_layer_size = 25    # 25 hidden units
num_labels = 10           # from 1 to 10 (mapped 10 to label 0)

# load data from mat file
data = loadmat(training_datafile)
