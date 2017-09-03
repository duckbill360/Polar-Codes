# Polar Codes with BP Decoding

import polar_codes
import numpy as np

############### PARAMETERS ###############
Times = 1
N = 1024        # code length
R = 0.5         # code rate
epsilon = 0.45   # cross-over probability for a BEC
SNR_in_db = 1.5

Var = 1 / (2 * pow(10.0, SNR_in_db / 10))
sigma = pow(Var, 1 / 2)

B_N = polar_codes.permutation_matrix(N)
frozen_indexes = polar_codes.generate_frozen_set_indexes(N, R, epsilon)
G = polar_codes.generate_G_N(N)

count = 0


def add_noise(signal, SNR):
    Var = 1 / (2 * R * pow(10.0, SNR_in_db / 10))
    sigma = pow(Var, 1 / 2)
    noise = np.random.normal(scale=sigma, size=(1, N))
    return signal + noise


def func():
    global count
    count += 1
    print(count)

    message = polar_codes.random_message_with_frozen_bits(N, R, epsilon, frozen_indexes)
    codeword = polar_codes.encode(message, G)
    signal = codeword * (-2) + 1
    signal = add_noise(signal, SNR_in_db)
    decoded_message = polar_codes.decode(signal, 60, frozen_indexes, B_N, sigma)
    error = (decoded_message != message) * 1.0

    return sum(error)


if __name__ == '__main__':
    # ################ ENCODING ################
    # print("------------------ENCODING------------------")
    #
    # message = polar_codes.random_message_with_frozen_bits(N, R, epsilon, frozen_indexes)
    # # Encode the message.
    # codeword = polar_codes.encode(message, G)
    #
    # # BPSK
    # # Mapping 0 => +1, 1 => -1
    # signal = codeword * (-2) + 1
    #
    # # Show the message and the corresponding codeword.
    # print('Message: \n', message)
    # print('Codeword:\n', codeword)
    # print('Signal:\n', signal)
    #
    #
    # ################# CHANNEL ################
    # signal = add_noise(signal, SNR_in_db)
    #
    #
    # ################ DECODING ################
    # print("------------------DECODING------------------")
    #
    # decoded_message = polar_codes.decode(signal, 60, frozen_indexes, B_N)
    #
    # print('Message: \n', message)
    # print('Decoded message: \n', decoded_message)
    #
    # error = (decoded_message != message) * 1.0
    # print('BER :', sum(error) / N)


    ################# TEST ###################
    import multiprocessing

    error_count = 0

    # Calling func() one time means doing encoding and decoding one time.

    multiprocessing.freeze_support()    # To avoid RunTimeError
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()  # Get the number of cpu cores

    ###########
    import timeit
    start = timeit.default_timer()

    results = []

    for j in range(Times):      # The number of iterations for each cpu core
        for i in range(0, cpus):        # The number of cpu cores
            result = pool.apply_async(func)
            results.append(result)

    pool.close()
    pool.join()

    for result in results:
        error_count += result.get()

    BER = error_count / (N * len(results))
    print('Error count :', error_count)
    print('Total bits :', N * len(results))
    print('BER :', BER)

    stop = timeit.default_timer()
    print('\nRun time :', (stop - start) // 60, 'minutes,', (stop - start) % 60, 'seconds')

