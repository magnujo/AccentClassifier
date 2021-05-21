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
