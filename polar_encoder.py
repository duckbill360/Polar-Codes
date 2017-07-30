# Polar Encoder Functions

import numpy as np
import math

def generate_G_N(N):

    F = np.array([1, 0],
                 [1, 1])

    return


# Generate the bit-reversed permutation matrix.
def permutation_matrix(N):
    R = np.zeros((N, N))

    for i in range(N):
        bit_reverse = int(bin(i)[2:].zfill(int(math.log2(N)))[::-1], 2)
        R[i][bit_reverse] = 1

    return R

