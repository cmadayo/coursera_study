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
