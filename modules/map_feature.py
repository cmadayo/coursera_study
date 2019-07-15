from typing import List
import numpy as np

# type define
Vector = List[float]
Matrix = List[Vector]

# description: this method can make too many features
# x1, x2 must be same size
def get_feature(x1: Vector, x2: Vector, degree: int) -> Matrix:
    # check columns and rows
    if x1.shape[0] != x2.shape[0]:
        print('x1, x2 must be same size')
        exit()

    return_feature = np.ones((x1.shape[0], 1))

    for i in range(1, degree+1):
        for j in range(0, i+1):
            insert_feature = np.reshape((x1**(i-j)) * (x2**j), (x1.shape[0], 1))
            return_feature = np.hstack((return_feature, insert_feature))

    return return_feature
