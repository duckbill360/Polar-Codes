# Polar Codes with BP Decoding

import polar_codes
import numpy as np

############### PARAMETERS ###############
Times = 10
N = 1024        # code length
R = 0.5         # code rate
epsilon = 0.45   # cross-over probability for a BEC
SNR_in_db = 1.5


################ ENCODING ################
print("------------------ENCODING------------------")

B_N = polar_codes.permutation_matrix(N)
frozen_indexes = polar_codes.generate_frozen_set_indexes(N, R, epsilon)
G = polar_codes.generate_G_N(N)
message = polar_codes.random_message_with_frozen_bits(N, R, epsilon, frozen_indexes)
# Encode the message.
codeword = polar_codes.encode(message, G)

# BPSK
# Mapping 0 => +1, 1 => -1
signal = codeword * (-2) + 1

# Show the message and the corresponding codeword.
print('Message: \n', message)
print('Codeword:\n', codeword)
print('Signal:\n', signal)


################# CHANNEL ################
def add_noise(signal, SNR):
    Var = 1 / (2 * R * pow(10.0, SNR_in_db / 10))
    sigma = pow(Var, 1 / 2)
    noise = np.random.normal(scale=sigma, size=(1, N))
    return signal + noise


################ DECODING ################
print("------------------DECODING------------------")
decoded_message = polar_codes.decode(signal, 60, frozen_indexes, B_N)

print('Message: \n', message)
print('Decoded message: \n', decoded_message)

error = (decoded_message != message).astype(np.float64)
print('BER :', sum(error) / N)


################# TEST ###################
error_count = 0
for i in range(Times):
    message = polar_codes.random_message_with_frozen_bits(N, R, epsilon, frozen_indexes)
    codeword = polar_codes.encode(message, G)
    signal = codeword * (-2) + 1
    signal = add_noise(signal, SNR_in_db)
    decoded_message = polar_codes.decode(signal, 60, frozen_indexes, B_N)
    error = (decoded_message != message).astype(np.float64)
    error_count += sum(error)

BER = error_count / (N * Times)
print('BER :', BER)