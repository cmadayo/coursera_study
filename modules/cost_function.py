from typing import List, Tuple
import numpy as np
from . import sigmoid

# type define
Vector = List[float]
Matrix = List[Vector]

def get_cost_grad(theta:Vector, X:Matrix, y:Vector) -> Tuple[float, Vector]:
    # check columns and rows
    if X.shape[1] != theta.shape[0]:
        print('X columns and theta rows must be same')
        exit()

    elif X.shape[0] != y.shape[0]:
        print('X and y must be same rows')
        exit()

    m = y.shape[0]

    # X:       100 * 3
    # theta:   3 * 1
    h = sigmoid.execute(X.dot(theta))      # 100 * 1
    h_sum = np.sum(-y*np.log(h) - (1-y)*np.log(1-h), axis=0)

    J = (1/m) * h_sum
    grad = (1/m) * X.T.dot((h - y))

    return (J, grad)


def get_reg_cost_grad(theta:Vector, X:Matrix, y:Vector, lambda_value:float) -> Tuple[float, Vector]:
    # check columns and rows
    if X.shape[1] != theta.shape[0]:
        print('X columns and theta rows must be same')
        exit()

    elif X.shape[0] != y.shape[0]:
        print('X and y must be same rows')
        exit()

    # size of y
    m = y.shape[0]

    # calculate non regularized value
    (J, grad) = get_cost_grad(theta, X, y)

    # theta for regularized param
    theta_reg = theta.copy()
    theta_reg[0] = 0

    # add regularized param
    J += lambda_value * np.sum(theta_reg**2, axis=0) / (2*m)
    grad += lambda_value * theta_reg / m

    return (J, grad)
