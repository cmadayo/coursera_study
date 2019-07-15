from typing import List
import numpy as np
import matplotlib.pyplot as plt

# type define
Vector = List[float]
Matrix = List[Vector]

# X must be 2 columns
# y must be 1 columns(vector)
# X and y must be same rows
def plot(X: List[Vector], y: Vector) -> None:
    # check columns and rows
    if X.shape[1] != 2:
        print('X must be 2 columns')
        exit()

    elif X.shape[0] != y.shape[0]:
        print('X and y must be same rows')
        exit()

    # find positive data and negative data
    positives = np.where(y == 1)
    negatives = np.where(y == 0)

    # plot data
    plt.plot(X[positives,0], X[positives,1], "b+")
    plt.plot(X[negatives,0], X[negatives,1], "yo")

    # show
    plt.show()
