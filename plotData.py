from typing import List
import numpy as np
import matplotlib.pyplot as plt

# type define
Vector = List[float]
Matrix = List[float]

# X must be 2 columns
# y must be 1 columns(vector)
# X and y must be same rows
def plot(X: List[Vector], y: Vector) -> None:
    # check columns and rows
    if X.ndim != 2:
        print('X must be 2 columns')
        exit()

    elif y.ndim != 1:
        print('y must be 1 columns')
        exit()

    elif X.shape[0] != y.shape[0]:
        print('X and y must be same rows')
        exit()

    # find positive data and negative data
    positives = np.where(y == 1)
    negatives = np.where(y == 0)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # plot data
    ax.scatter(X[positives,0], X[positives,1], c='red')
    ax.scatter(X[negatives,0], X[negatives,1], c='blue')

    # show
    plt.show()
