# Polar Codes with BP Decoding
#

import polar_codes
import numpy as np

################ ENCODING ################
print("------------------ENCODING------------------")

message = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
# G is the generator matrix and the size is specified by the message length.
G = polar_codes.generate_G_N(message.size)
print('Generator Matrix:\n', G)
codeword = polar_codes.encode_message(message)

print('Message: \n', message)
print('Codeword:\n', codeword)


################ DECODING ################
print("\n\n------------------DECODING------------------")


