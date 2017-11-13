# Polar Encoder Functions

import numpy as np
from functools import lru_cache


################## CHANNEL POLARIZATION ##################
# This function determines the frozen bit positions.
# Careful: i and N should be no less than 1.
@lru_cache(maxsize=None)
def Z_W(i, N, eps):
    # The root node index is i = 1, N = 1 and the value is eps.
    if i == 1 and N == 1:
        return eps
    # Have not reached the root yet.
    elif N > 1:
        # The left node
        if i % 2 == 1:
            return 2 * Z_W((i + 1) // 2, N // 2, eps) - pow(Z_W((i + 1) // 2, N // 2, eps), 2)
        # The right node
        elif i % 2 == 0:
            return pow(Z_W(i // 2, N // 2, eps), 2)


######################## ENCODING ########################
# This function generates the generator matrix of size N
def generate_G_N(N):
    n = int(np.log2(N))
    F_2 = np.array([[1, 0],
                    [1, 1]])
    B_N = permutation_matrix(N)     # B_N is a symmetric

    # The nth Kronecker product of F_2
    # Initializing this matrix to a "2D" matrix with only 1 element
    nth_Kronecker_product = np.array([[1]])
    for i in range(n):
        nth_Kronecker_product = np.kron(nth_Kronecker_product, F_2)

    # Return B_N * F_2**n
    return np.dot(B_N, nth_Kronecker_product) % 2


def random_message_with_frozen_bits(N, R, epsilon, frozen_indexes):
    # Randomly generate a message vector of size N.
    message = np.random.randint(2, size=N)
    message = message.astype(np.float64)  # convert dtype to float64

    for i in frozen_indexes:
        message[i] = 0

    return message


# This function tells the positions (indexes) of the frozen bit.
def generate_frozen_set_indexes(N, R, epsilon):
    U = [Z_W(i + 1, N, eps=epsilon) for i in range(N)]

    U_sorted = U.copy()     # Use .copy() to actually "copy" the list
    U_sorted.sort(reverse=True)  # Descending order

    index = int(N * R - 1)
    threshold = U_sorted[index]

    index_set = []
    for i in range(N):
        if U[i] >= threshold:
            index_set.append(i)

    return index_set


# The function generates the bit-reversed permutation matrix.
def permutation_matrix(N):
    # Initializing this matrix to an N-by-N all-zero matrix
    R = np.zeros((N, N))

    for i in range(N):
        bit_reversed_int = int(bin(i)[2:].zfill(int(np.log2(N)))[::-1], 2)
        R[i][bit_reversed_int] = 1

    return R


# This function computes the codeword: x = u * G_N
# "message" should be a 1D (row) vector.
def encode(message, G):
    # N is the length of the message
    N = message.size

    codeword = np.dot(message, G) % 2       # Mod 2 to convert to the binary vector.
    # "codeword" should also be a 1D (row) vector

    return codeword


######################## DECODING ########################
# Belief-Propagation Decoding
# "x" is a 1D vector.
def decode(x, iteration_num, frozen_set_indexes, B_N, sigma, alpha, beta):
    N = x.size                # code length
    n = int(np.log2(N))       # log2(N)
    L = np.zeros((N, n + 1))  # L message array
    R = np.zeros((N, n + 1))  # R message array

    # Initialization
    LLR_L = x * 2 / pow(sigma, 2)
    LLR_R = np.zeros(N)       # 1D all-zero vector

    for i in frozen_set_indexes:
        LLR_R[i] = 1000000    # Set every element of index in frozen_set_indexes to 1000

    # Bit-reversed permutation
    LLR_R = np.dot(LLR_R, B_N)

    L[:, n] = LLR_L     # L[:, n] is the channel side
    R[:, 0] = LLR_R     # R[:, 0] is the user side
    # These 2 can not be changed during iteration.
    # END of initialization

    # Message-Passing Algorithm
    for k in range(iteration_num):

        ############ R propagation
        # R[, 0] shouldn't be overwritten.
        for i in range(n - 1):  # 1 stage fewer than L propagation
            R[: N: 2, i + 1] = f(R[: N // 2, i], R[N // 2:, i] + L[1: N: 2, i + 1])
            R[1: N: 2, i + 1] = f(L[: N: 2, i + 1], R[: N // 2, i]) + R[N // 2:, i]

        ############ L propagation
        # j moves vertically, i moves horizontally
        # We can ignore the L[:, 0] here.
        for i in range(n - 1, -1, -1):      # i is the width counted from the left  (n-1)~0
            L[:N // 2, i] = f(L[: N: 2, i + 1], L[1: N: 2, i + 1] + R[N // 2:, i])
            L[N // 2:, i] = f(R[: N // 2, i], L[: N: 2, i + 1]) + L[1: N: 2, i + 1]

            ########################################################################
            # These lines of code are for the "Expediting" BP decoder.
            # if i == n - 1:
            #     L[:N // 2, i] = alpha[k, :N // 2] * L[:N // 2, i] + beta[k, N // 2:] * R[N // 2:, i]
            #     L[N // 2:, i] = alpha[k, N // 2:] * L[N // 2:, i] + beta[k, :N // 2] * R[:N // 2, i]
            ########################################################################

    output = np.dot(L[:, 0], B_N)       # Permutation
    message = output < 0
    message = message.astype(np.float64)

    # Force the frozen bits to be '0's.
    for i in frozen_set_indexes:
        message[i] = 0

    return message


# x and y are floating-point numbers
def f(x, y):
    return np.sign(x) * np.sign(y) * np.minimum(np.absolute(x), np.absolute(y))
