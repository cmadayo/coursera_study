import numpy as np
from scipy.io import loadmat
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
X = data['X']                               # 5000 * 400
y = np.reshape(data['y'],(-1,))             # 5000 * 1

m = X.shape[0]

# select random training set
random_array = np.random.permutation(m)
selected_array = X[random_array[0:100], :]       # 100 * 400

# display data from mat
#plot_data.mat_data_plot(selected_array, 20, 20)  # 2nd 3rd param -> 20x20 Input Images of Digits

theta_temp = np.array([-2, -1, 1, 2])
X_temp = np.c_[np.ones((5,1)), np.reshape(np.arange(1, 16), (5, 3), 'F')/10]  # 5 * 4   ...[1 0.1 ~ 1.5] Matrix
y_temp = (np.array([1, 0, 1, 0, 1]) >= 0.5).astype(np.int)          # 5 * 1    boolean to binary
lambda_temp = 3

# # ---------------- hand made function start ----------------
# comment out cuz too slow....
# (J, grad) = cost_function.get_reg_cost_grad(theta_temp, X_temp, y_temp, lambda_temp)
# print('My function Result: cost = {}, gradients = {}'.format(J, grad))
# lambda_value = 0.1
# all_theta = one_vs_all.get_theta(X, y, num_labels, lambda_value)
# ---------------- hand make function end ----------------

# library costFunction
lr = LogisticRegression(multi_class="ovr", solver="newton-cg")
lr.fit(X, y)
print('y, predictX= {}'.format(np.c_[y, lr.predict(X)]))
print(lr.score(X, y))
