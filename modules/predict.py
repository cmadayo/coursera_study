from typing import List, Tuple
import numpy as np
from . import sigmoid

# type define
Vector = List[float]
Matrix = List[Vector]

def execute(theta: Vector, X: Matrix, threshold: float) -> Vector:
    if X.shape[1] != theta.shape[0]:
        print('X columns and theta rows must be same')
        exit()

    p = sigmoid.execute(X.dot(theta)) >= threshold
    return p

# Theta1's rows num + 1 and Theta2's colomns num must be same
# X's columns num + 1 and Theta1's columns num must be same
def execute_for_neural_network(Theta1:Matrix, Theta2:Matrix, X:Matrix) -> Vector:
    if Theta1.shape[0] + 1 != Theta2.shape[1]:
        print("Theta1's rows num + 1 and Theta2's colomns num must be same")
        exit()

    if X.shape[1] + 1 != Theta1.shape[1]:
        print("X's columns num + 1 and Theta1's columns num must be same")
        exit()

    m = X.shape[0]                   # training set size
    num_labels = Theta2.shape[0]     # output layer size

    # forward propagation
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1).T  # 401 * 5000
    a2 = sigmoid.execute(Theta1.dot(a1))                 # 25 * 5000
    a2 = np.concatenate([np.ones((1, m)), a2])           # 26 * 5000
    a3 = sigmoid.execute(Theta2.dot(a2))                 # 10 * 5000

    result = np.argmax(a3, axis=0) + 1  # add + 1 cuz matlab format is 0~9 to 1~10(0 == 10)

    return result
