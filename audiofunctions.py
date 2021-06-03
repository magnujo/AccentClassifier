from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_io as tfio


def collect_all_amplitudes(audio_data_dir_path: str):
    final = np.array([])
    sum = 0
    pathlist = Path(audio_data_dir_path).rglob('*')
    for path in pathlist:
        path_in_str = str(path.resolve())
        print(path_in_str)
        audio = tfio.audio.AudioIOTensor(path_in_str)
        audio_tensor = tf.squeeze(audio.to_tensor(), axis=[-1])
        tensor = tf.cast(audio_tensor, tf.float32)
        tensor_np = tensor.numpy()
        final = np.append(final, tensor_np)
        sum += tensor_np.shape[0]
    assert sum == final.shape[0]
    return final


'''
Takes a path to an audio folder, and draws a histogram of the distribution of amplitudes across all the audio files.
'''


def draw_amplitude_distribution(audio_data_dir_path: str):
    amplitudes = collect_all_amplitudes(audio_data_dir_path)
    plt.hist(amplitudes, bins=15, edgecolor="Black", log=True)
    plt.show()


'''
Da der ikke er plads til intervallerne på x-asken bliver tegningen lidt rodet
'''


def draw_frequency_distribution(audio_data_dir_path: str, n_bins):
    freq_array = collect_all_amplitudes_of_frequencies(audio_data_dir_path)
    plot_array_of_freq_amps(freq_array, n_bins)


'''
Goes through a folder of audio files and sums up all the amplitudes of the different frequencies (seems like its usually
from 0hz - 257hz. Maybe it is just fixed to 257 and then the files that have higher frequencies are compressed/filtered?
Needs to be tested on more files). All of the frequency amplitudes of mp3 files are apparently negative, 
and some of the wavs are also. Does this make sense? Might want to change the spectogram generation method
to something similar to train_with_padding.py method.

'''


# %%
def collect_all_amplitudes_of_frequencies(audio_data_dir_path: str):
    final = np.array([])
    path_list = Path(audio_data_dir_path).rglob('*wav')  # r = recursive (all subfolders)
    for path in path_list:
        path_in_str = str(path.resolve())
        audio = tfio.audio.AudioIOTensor(path_in_str)
        audio_tensor = tf.squeeze(audio.to_tensor(), axis=[-1])
        tensor = tf.cast(audio_tensor, tf.float32)
        spectrogram = tfio.audio.spectrogram(tensor, nfft=512, window=512, stride=256)
        spectrogram = tf.math.log(spectrogram).numpy()
        spectrogram = np.where(spectrogram == float("-inf"), 0,
                               spectrogram)  # replaces -inf with 0, so the below sum works.
        sum_of_freqamps = np.sum(spectrogram, axis=0)  # Sums up the frequency amplitudes
        final = add_arrays_of_different_size(sum_of_freqamps,
                                             final)  # all of the files i tested on has the same size, so maybe not needed? Needs to be tested on more files.

    return final


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    return audio


'''
    Sums up the amplitudes of frequencies of a single file, and returns it as an array where the index is the frequency. For example
    sum_of_freqamps[1] = sum of 1hz amplitudes
'''


def single_freq_hist(file_path: str):
    path_in_str = file_path
    audio = tf.io.read_file(path_in_str)
    audio = decode_audio(audio)
    audio = tf.cast(audio, tf.float32)
    spectrogram = tfio.audio.spectrogram(audio, nfft=512, window=512, stride=256)
    spectrogram = tf.math.log(spectrogram).numpy()
    spectrogram = np.where(spectrogram == float("-inf"), 0,
                           spectrogram)  # replaces -inf with 0, so the below sum works.
    sum_of_freqamps = np.sum(spectrogram, axis=0)  # Sums up the frequency amplitudes
    return sum_of_freqamps


def add_arrays_of_different_size(a, b):
    if len(a) == 0:
        c = b
    elif len(b) == 0:
        c = a
    elif len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c


'''
Plotting with intervals on the x axis 
'''


def plot_array_of_freq_amps(x, n_bins):
    freqinterval = len(x) / n_bins
    split = np.array_split(x, n_bins)
    print("split", split)
    b = list(map(lambda y: np.sum(y), split))
    print("b", b)
    d = {}
    for i in range(n_bins):
        k = str(i * freqinterval) + "-" + str((i * freqinterval) + freqinterval)
        d[k] = b[i]
    print("d", d)
    return d


'''
Plotting with i on the x axis 
'''


def plot_array_of_freq_amps2(x, n_bins):
    freqinterval = len(x) / n_bins
    split = np.array_split(x, n_bins)
    print("split", split)
    b = list(map(lambda y: np.sum(y), split))
    print("b", b)
    d = {}
    for i in range(n_bins):
        k = str(i * freqinterval) + "-" + str((i * freqinterval) + freqinterval)
        d[i] = b[i]
    print("d", d)
    return d


if __name__ == "__main__":
    draw_amplitude_distribution(r"D:\data_small\cv-corpus-6.1-2020-12-11\en\wav")

    print("finish")
