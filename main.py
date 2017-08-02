# Polar Codes with BP Decoding

import polar_codes
import numpy as np

############### PARAMETERS ###############
N = 8      # code length
R = 0.5    # code rate
epsilon = 0.5   # cross-over probability for a BEC



################ ENCODING ################
print("------------------ENCODING------------------")

message = polar_codes.random_message_with_frozen_bits(N, R, epsilon)
U = polar_codes.generate_frozen_set_indexes(N, R, epsilon)
# Encode the message.
codeword = polar_codes.encode(message)
signal = codeword * (-2) + 1

# Show the message and the corresponding codeword.
print('Message: \n', message)
print('Codeword:\n', codeword)
print('Signal:\n', signal)

################# CHANNEL ################




################ DECODING ################
print("------------------DECODING------------------")
polar_codes.decode(signal, 1, U)

