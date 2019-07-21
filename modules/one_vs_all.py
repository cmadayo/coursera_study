from typing import List, Tuple
import numpy as np
from . import cost_function
from scipy.optimize import fmin_bfgs

# type define
Vector = List[float]
Matrix = List[Vector]

# X rows and y rows num must be same
def get_theta(X: Matrix, y:Vector, num_labels:List[int], lambda_value:float) -> Vector:
    if X.shape[0] != y.shape[0]:
        print('X rows and y rows num must be same')
        exit()

    m = X.shape[0]
    n = X.shape[1]

    all_theta = np.zeros((num_labels, n+1))
    X = np.c_[np.ones((m,1)), X]

    for num in range(1, num_labels+1):
        num %= 10
        print(num)
        initial_theta = np.zeros((n+1,))
        # wrapper for fmin
        cost_function_reg_wrapper = lambda theta, X, y, lambda_value: cost_function.get_reg_cost_grad(theta, X, y, lambda_value)[0]

        # minimize costFunction's cost
        theta = fmin_bfgs(cost_function_reg_wrapper, initial_theta, args=(X, (y == num).astype(np.int), lambda_value,), full_output=True, disp=True)[0]

        # minimize costFunction's cost result
        print(theta)
        all_theta[num] = theta.copy()

    return all_theta
