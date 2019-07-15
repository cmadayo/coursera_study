from typing import List
import numpy as np

# type define
Vector = List[float]

def execute(z: Vector) -> Vector:
    return 1 / (1 + np.exp(-z))
