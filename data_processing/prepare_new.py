import numpy as np
from tools import *
from create_vector_headpose import pad_sequence, create_vectors

N_CONTEXT = 60
N_INPUT = 26

# input_vectors = calculate_mfcc('new_data/audio5.wav')
# Step 4: Retrieve N_CONTEXT each time, stride one by one
# input_with_context = np.array([])
# output_with_context = np.array([])

# strides = len(input_vectors)

# input_vectors = pad_sequence(input_vectors)

# for i in range(strides):
#     stride = i + int(N_CONTEXT/2)
#     if i == 0:
#         input_with_context = input_vectors[stride - int(N_CONTEXT/2) : stride + int(N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT)
#         # output_with_context = output_vectors[i].reshape(1, N_OUTPUT)
#     else:
#         input_with_context = np.append(input_with_context, input_vectors[stride - int(N_CONTEXT/2) : stride + int(N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT), axis=0)
#         # output_with_context = np.append(output_with_context, output_vectors[i].reshape(1, N_OUTPUT), axis=0)

# # return input_with_context, output_with_context

# np.save('new_data/audio5.npy', input_with_context)

i, o = create_vectors('obama_processed/train/inputs/audio32.wav',
               'obama_processed/train/labels/pose32.csv')

mkdir -p 'new_data'
np.save('new_data/X32.npy', i)
np.save('new_data/Y32.npy', o)

