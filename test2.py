import math

import librosa
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from sklearn.preprocessing import MinMaxScaler

DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10  # 35#250
frame_length = 1024
spect_length = int(frame_length / 2 + 1)
step_time = 0.008
frame_step = int(RATE * step_time)


@tf.function
def get_wav(path):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''
    binary = tf.io.read_file(path)
    wav, sample_rate = tf.audio.decode_wav(binary, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    #sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    #wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC)


def remove_silence(wav, thresh=0.001, chunk=5000):
    '''
    Searches wav form for segments of silence. If wav form values are lower than 'thresh' for 'chunk' samples, the values will be removed
    :param wav (np array): Wav array to be filtered
    :return (np array): Wav array with silence removed
    '''

    return wav[(wav >= thresh) & (wav <= -thresh)]


def normalize_mfcc(mfcc):
    '''
    Normalize mfcc
    :param mfcc:
    :return:
    '''
    mms = MinMaxScaler()
    return mms.fit_transform(np.abs(mfcc))


def pre_process(path):
    wav = get_wav(path)
    # wav = remove_silence(wav)
    wav = to_mfcc(wav)
    wav = normalize_mfcc(wav)
    wav = np.expand_dims(wav, -1)
    wav = tf.convert_to_tensor(wav)
    wav = tf.image.resize(wav, (100, 100))
    return wav
