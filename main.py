# Polar Codes with BP Decoding

import polar_codes
import numpy as np

############### PARAMETERS ###############
N = 8   # code length
epsilon = 0.5   # cross-over probability for a BEC
R = 0.5    # code rate


################ ENCODING ################
print("------------------ENCODING------------------")

# Randomly generate a message vector of size N.
message = np.random.randint(2, size=N)
message = message.astype(np.float64)    # convert dtype to float64

# Determine the frozen set.
U = [polar_codes.Z_W(i + 1, N, eps=epsilon) for i in range(N)]
# The parameter i, N for W_N() should be no less than 1
print(U)

# Encode the message.
codeword = polar_codes.encode_message(message)

# Show the message and the corresponding codeword.
print('Message: \n', message)
print('Codeword:\n', codeword)



################# CHANNEL ################




################ DECODING ################
print("\n\n------------------DECODING------------------")


