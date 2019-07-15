from typing import List
import numpy as np
import matplotlib.pyplot as plt

# type define
Vector = List[float]
Matrix = List[Vector]

# X must be 2 columns
# X and y must rows be same num
def scatter_plot(X: List[Vector], y: Vector) -> None:
    # check columns and rows
    if X.shape[1] != 2:
        print('X must be 2 columns')
        exit()

    elif X.shape[0] != y.shape[0]:
        print('X and y must rows be same num')
        exit()

    # find positive data and negative data
    positives = np.where(y == 1)
    negatives = np.where(y == 0)

    # plot data
    plt.plot(X[positives,0], X[positives,1], "b+")
    plt.plot(X[negatives,0], X[negatives,1], "yo")

    # show
    plt.show()

# X columns and theta rows must be same num
# X and y must be same rows num
def scatter_plot_border(theta:Vector, X:Matrix, y:Vector) -> None:
    # check columns and rows
    if X.shape[1] != theta.shape[0]:
        print('X columns and theta rows must be same num')
        exit()

    elif X.shape[0] != y.shape[0]:
        print('X and y rows must be same num')
        exit()

    plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x, plot_y)
    scatter_plot(X[:, 1:3], y)
