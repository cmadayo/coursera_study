from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotData

# type define
Vector = List[float]
Matrix = List[Vector]

def plot(theta:Vector, X:Matrix, y:Vector) -> None:
    # check columns and rows
    if X.shape[1] != theta.shape[0]:
        print('X columns and theta rows must be same')
        exit()

    elif X.shape[0] != y.shape[0]:
        print('X and y must be same rows')
        exit()

    plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x, plot_y)
    plotData.plot(X[:, 1:3], y)
