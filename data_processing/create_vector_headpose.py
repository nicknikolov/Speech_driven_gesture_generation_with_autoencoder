
"""
This script does preprocessing of the dataset specified in DATA_DIR
 and stores it in the same folder as .npy files
It should be used before training, as described in the README.md

@author: Taras Kucherenko
(remix by nick)
"""

import os
import sys
import pandas as pd

import pyquaternion as pyq
from scipy.spatial.transform import Rotation as R

from tools import *

# N_OUTPUT = 192 * 2 # Number of gesture features (position)
N_OUTPUT = 12 # translation, look direction and their velocities
WINDOW_LENGTH = 50 # in miliseconds
FEATURES = "MFCC"
N_CONTEXT = 60

if FEATURES == "MFCC":
    N_INPUT = 26 # Number of MFCC features

def pad_sequence(input_vectors):
    """
    Pad array of features in order to be able to take context at each time-frame
    We pad N_CONTEXT / 2 frames before and after the signal by the features of the silence
    Args:
        input_vectors:      feature vectors for an audio

    Returns:
        new_input_vectors:  padded feature vectors
    """

    if FEATURES == "MFCC":
        # Pad sequence not with zeros but with MFCC of the silence
        silence_vectors = calculate_mfcc("data_processing/silence.wav")
        mfcc_empty_vector = silence_vectors[0]

        empty_vectors = np.array([mfcc_empty_vector] * int(N_CONTEXT / 2))


    # append N_CONTEXT/2 "empty" mfcc vectors to past
    new_input_vectors = np.append(empty_vectors, input_vectors, axis=0)
    # append N_CONTEXT/2 "empty" mfcc vectors to future
    new_input_vectors = np.append(new_input_vectors, empty_vectors, axis=0)

    return new_input_vectors

def create_vectors(audio_filename, gesture_filename):
    """
    Extract features from a given pair of audio and motion files
    Args:
        audio_filename:    file name for an audio file (.wav)
        gesture_filename:  file name for a motion file (.bvh)

    Returns:
        input_with_context   : speech features
        output_with_context  : motion features
    """
    # Step 1: Vactorizing speech, with features of N_INPUT dimension, time steps of 0.01s
    # and window length with 0.025s => results in an array of 100 x N_INPUT

    if FEATURES == "MFCC":
        input_vectors = calculate_mfcc(audio_filename)

    data = pd.read_csv(gesture_filename, sep=', ', engine='python')

    Ts = data[['pose_Tx', 'pose_Ty', 'pose_Tz']].to_numpy()
    # since data was sources on different camera angles, lets assume face is always in the centre
    mean_pose = Ts.mean(axis=(0))
    Ts = Ts - mean_pose

    Rs = data[['pose_Rx', 'pose_Ry', 'pose_Rz']].to_numpy()

    # OpenFace convention is positive Z to be away from camera
    # I assume face is facing camera, so direction is towards negative z
    Ts_dir = Ts.copy()
    Ts_dir[:, 2] -= 50

    for i in range(Ts_dir.shape[0]):
        t = Ts[i, :]
        dir = Ts_dir[i, :] - t
        r = Rs[i, :]

        rot = R.from_rotvec(r)
        new_dir = rot.apply(dir) + t
        Ts_dir[i, :] = new_dir

    # remove every third element from my headpose to get to 20fps
    # print('Ts', Ts.shape)

    # Ts = Ts[::3, :]
    Ts = np.delete(Ts, slice(None, None, 3), axis=0)
    Ts_dir = np.delete(Ts_dir, slice(None, None, 3), axis=0)

    # calculate velocity
    T_vels = []
    T_vels.append([0, 0, 0])
    t_prev = Ts[0]
    for t in Ts[1:-1]:
        vel = t - t_prev
        T_vels.append(vel)
    T_vels.append([0, 0, 0])

    T_dir_vels = []
    T_dir_vels.append([0, 0, 0])
    t_prev = Ts_dir[0]
    for t in Ts_dir[1:-1]:
        vel = t - t_prev
        T_dir_vels.append(vel)
    T_dir_vels.append([0, 0, 0])

    output_vectors = np.concatenate((Ts, Ts_dir, T_vels, T_dir_vels), axis=1)

    # # Step 3: Align vector length
    input_vectors, output_vectors = shorten(input_vectors, output_vectors)

    # Step 4: Retrieve N_CONTEXT each time, stride one by one
    input_with_context = np.array([])
    output_with_context = np.array([])

    strides = len(input_vectors)

    input_vectors = pad_sequence(input_vectors)

    for i in range(strides):
        stride = i + int(N_CONTEXT/2)
        if i == 0:
            input_with_context = input_vectors[stride - int(N_CONTEXT/2) : stride + int(N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT)
            output_with_context = output_vectors[i].reshape(1, N_OUTPUT)
        else:
            input_with_context = np.append(input_with_context, input_vectors[stride - int(N_CONTEXT/2) : stride + int(N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT), axis=0)
            output_with_context = np.append(output_with_context, output_vectors[i].reshape(1, N_OUTPUT), axis=0)

    return input_with_context, output_with_context

def create(name):
    """
    Create a dataset
    Args:
        name:  dataset: 'train' or 'test' or 'dev

    Returns:
        nothing: saves numpy arrays of the features and labels as .npy files

    """
    DATA_FILE = pd.read_csv(DATA_DIR + '/gg-' + str(name) + '.csv')
    X = np.array([])
    Y = np.array([])

    for i in range(len(DATA_FILE)):
        input_vectors, output_vectors = create_vectors(DATA_FILE['wav_filename'][i], DATA_FILE['pose_filename'][i])

        if len(X) == 0:
            X = input_vectors
            Y = output_vectors
        else:
            X = np.concatenate((X, input_vectors), axis=0)
            Y = np.concatenate((Y, output_vectors), axis=0)

        if i%3==0:
            print("^^^^^^^^^^^^^^^^^^")
            print('{:.2f}% of processing for {:.8} dataset is done'.format(100.0 * (i+1) / len(DATA_FILE), str(name)))
            print("Current dataset sizes are:")
            print(X.shape)
            print(Y.shape)

    x_file_name = DATA_DIR + '/X_' + str(name) + '.npy'
    y_file_name = DATA_DIR + '/Y_' + str(name) + '.npy'
    np.save(x_file_name, X)
    np.save(y_file_name, Y)


def create_test_sequences(dataset):
    """
    Create test sequences
    Args:
        dataset:  dataset name ('train', 'test' or 'dev')

    Returns:
        nothing, saves dataset into .npy file

    """
    DATA_FILE = pd.read_csv(DATA_DIR + '/gg-'+dataset+'.csv')

    print('DATA DIR', DATA_DIR)
    print('DATA DIR', dataset)

    for i in range(len(DATA_FILE)):
        input_vectors, output_vectors = create_vectors(DATA_FILE['wav_filename'][i], DATA_FILE['pose_filename'][i])


        array = DATA_FILE['wav_filename'][i].split("/")
        name = array[len(array)-1].split(".")[0]

        X = input_vectors
        Y = output_vectors

        if not os.path.isdir(DATA_DIR + '/'+dataset+'_inputs'):
            os.makedirs(DATA_DIR +  '/'+dataset+'_inputs')

        x_file_name = DATA_DIR + '/'+dataset+'_inputs/X_test_' + name + '.npy'
        y_file_name = DATA_DIR + '/'+dataset+'_inputs/Y_test_' + name + '.npy'

        np.save(x_file_name, X)
        np.save(y_file_name, Y)


if __name__ == "__main__":

    # Check if script get enough parameters
    if len(sys.argv) < 3:
        raise ValueError('Not enough paramters! \nUsage : python ' + sys.argv[0].split("/")[-1] + ' DATA_DIR N_CONTEXT')

    # Check if the dataset exists
    if not os.path.exists(sys.argv[1]):
        raise ValueError(
            'Path to the dataset ({}) does not exist!\nPlease, provide correct DATA_DIR as a script parameter'
            ''.format(sys.argv[1]))

    DATA_DIR = sys.argv[1]
    N_CONTEXT = int(sys.argv[2])

    create_test_sequences('test')

    create('test')
    create('dev')
    create('train')
