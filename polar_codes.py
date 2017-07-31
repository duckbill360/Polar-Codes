# Polar Encoder Functions

import numpy as np


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
    codeword = np.dot(message, G) % 2
    # "codeword" should also be a 1D (row) vector

    return codeword


######################## DECODING ########################

