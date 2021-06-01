import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt
def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def add_padding(waveform):
    waveform = tf.cast(waveform, tf.float32)
    zero_padding = tf.zeros([100_000] - tf.shape(waveform), dtype=tf.float32)
    return tf.concat([waveform, zero_padding], 0)

def get_spectrogram(waveform):
    waveform = add_padding(waveform)
    spectrogram = tfio.audio.spectrogram(waveform, nfft=512, window=512, stride=256)
    spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


if __name__ == "__main__":
    wav = decode_audio(r"D:\data_small\cv-corpus-6.1-2020-12-11\en\wav\common_voice_en_159722.wav")
    spectogram = get_spectrogram(wav)


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
    timescale = np.arange(wav.shape[0])
    axes[0].plot(timescale, wav.numpy())
    axes[0].set_title('Waveform')
    # axes[0].set_xlim([0, 16000])
    plot_spectrogram(spectogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram (With Padding)')
    plt.show()