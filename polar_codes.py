# Polar Encoder Functions

import numpy as np
from functools import lru_cache


################## CHANNEL POLARIZATION ##################
# This function determines the frozen bit positions.
# Careful: i and N should be no less than 1.
@lru_cache(maxsize=None)
def Z_W(i, N, eps):
    if i == 1 and N == 1:
        return eps
    elif N > 1:
        if i % 2 == 1:
            return 2 * Z_W((i + 1) // 2, N // 2, eps) - pow(Z_W((i + 1) // 2, N // 2, eps), 2)
        elif i % 2 == 0:
            return pow(Z_W(i // 2, N // 2, eps), 2)



######################## ENCODING ########################
# This function generates the generator matrix of size N
def generate_G_N(N):
    n = int(np.log2(N))
    F_2 = np.array([[1, 0],
                    [1, 1]])
    B_N = permutation_matrix(N)

    # The nth Kronecker product of F_2
    # Initializing this matrix to a "2D" matrix with only 1 element
    nth_Kronecker_product = np.array([[1]])
    for i in range(n):
        nth_Kronecker_product = np.kron(nth_Kronecker_product, F_2)

    # Return B_N * F_2**n
    return np.dot(B_N, nth_Kronecker_product) % 2


# The function generates the bit-reversed permutation matrix.
def permutation_matrix(N):
    # Initializing this matrix to an N-by-N all-zero matrix
    R = np.zeros((N, N))

    for i in range(N):
        bit_reverse = int(bin(i)[2:].zfill(int(np.log2(N)))[::-1], 2)
        R[i][bit_reverse] = 1

    return R


# This function computes the codeword: x = u * G_N
# "message" should be a 1D (row) vector.
def encode_message(message):
    # N is the length of the message
    N = message.size
    G = generate_G_N(N)

    # Show the generator matrix.
    print('Generator Matrix:\n', G)

    codeword = np.dot(message, G) % 2
    # "codeword" should also be a 1D (row) vector

    return codeword


######################## DECODING ########################



