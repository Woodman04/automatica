import numpy as np

class MatrixUtils():
    @staticmethod
    def is_square(matrix : np.array):
        return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    

 
    