#Most of this is from https://www.tensorflow.org/tutorials/audio/simple_audio

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# %%
#from https://www.tensorflow.org/tutorials/audio/simple_audio
#converts the binary wav file to a tensor waveform
def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


# %%
'''
Den her er mærkelig med dens zero_padding. Hvis zero padding er slået til får man exception.
Vi skal finde ud af hvor lang det længste lydklip er og så padde resten så de 
blive lige så lange. Lige nu padder den ud fra at alle filer er <1 sekund (16000 zeros) (så den padder 
med (1sekund-længden af lydfilen)), hvilket fucker op hvis filerne er over 1 sekund da det giver en minus padding,
som den ikke kan finde ud af. 
Men hvordan finder vi ud af hvor lang den længste fil er? 
'''
def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    #equal_length = tf.concat([waveform, zero_padding], 0)
    equal_length = tf.concat([waveform], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram





#%%
#Creates a audio player (only works in jupiter possibly)
display.display(display.Audio(wf, rate=16000))

#%%
data_dir = Path('data/audio_files')
print(data_dir.resolve())

#%%
#Filerne ligges i ./data/audio_files
wf = decode_audio(str(data_dir.resolve()) + "\\common_voice_en_39978.wav")
print(wf)
print("length")
print(tf.shape(wf))
plt.plot(wf)
plt.show()

#%%
filenames = tf.io.gfile.glob(str(data_dir) + "/*.wav")
filenames = tf.random.shuffle(filenames)

print(filenames)

#%%
spectrogram = get_spectrogram(wf)

#%%
def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(wf.shape[0])
axes[0].plot(timescale, wf.numpy())
axes[0].set_title('Waveform')
#axes[0].set_xlim([0, 16000])
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()
