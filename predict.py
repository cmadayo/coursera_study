from typing import List, Tuple
import numpy as np
import sigmoid

# type define
Vector = List[float]
Matrix = List[Vector]

def execute(theta: Vector, X: Matrix) -> Vector:
    if X.shape[1] != theta.shape[0]:
        print('X columns and theta rows must be same')
        exit()

    m = X.shape[0]
    p = sigmoid.execute(X.dot(theta)) >= 0.5
    return p
