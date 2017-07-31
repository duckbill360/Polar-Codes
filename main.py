# Polar Codes with BP Decoding
#

import polar_codes
import numpy as np

# G is the generator matrix
G = polar_codes.generate_G_N(8)
print(G)

message = np.array([1, 0, 1, 0, 1, 0, 1, 0])
codeword = polar_codes.encode(message)

print(codeword)

