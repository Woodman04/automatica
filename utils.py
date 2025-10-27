import numpy as np

class Utils():
    @staticmethod
    def is_square(matrix):
        return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
 
    