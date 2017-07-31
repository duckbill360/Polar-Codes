# Polar Encoder Functions

import numpy as np
import math


def generate_G_N(N):

    n = int(math.log2(N))
    F_2 = np.array([[1, 0],
                    [1, 1]])
    B_N = permutation_matrix(N)

    nth_Kronecker_product = np.array([[1]], dtype=np.float64)
    for i in range(n):
        nth_Kronecker_product = np.kron(nth_Kronecker_product, F_2)

    print(nth_Kronecker_product)

    return np.dot(B_N, nth_Kronecker_product)


# Generate the bit-reversed permutation matrix.
def permutation_matrix(N):
    R = np.zeros((N, N))

    for i in range(N):
        bit_reverse = int(bin(i)[2:].zfill(int(math.log2(N)))[::-1], 2)
        R[i][bit_reverse] = 1

    return R

