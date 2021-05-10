import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)
from tfx import tools
df = pd.read_csv(r"D:\data\cv-corpus-6.1-2020-12-11\en\df_accent.csv")

# unique = df["accent"].unique()
path = Path(r"D:\data\cv-corpus-6.1-2020-12-11\en\wav")
df["path"] = df["path"].apply(lambda x: str(path.joinpath(x.replace("mp3", "wav"))))
df = df.sample(frac=1)

jobs_encoder = LabelBinarizer()
jobs_encoder.fit(df['accent'])
transformed = jobs_encoder.transform(df['accent'])
ohe_df = pd.DataFrame(transformed)

X_features = df["path"].to_numpy()
Y_labels = ohe_df.to_numpy()



def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    return audio


def get_wav_and_label(file_name: tf.string, label):
    audio_binary = tf.io.read_file(file_name)
    wav = decode_audio(audio_binary)
    return wav, label


X_features = tf.convert_to_tensor(X_features, dtype=tf.string)
Y_labels = tf.convert_to_tensor(Y_labels, dtype=tf.float32)

X_features_ds = tf.data.Dataset.from_tensor_slices(X_features)
Y_labels_ds = tf.data.Dataset.from_tensor_slices(Y_labels)

AUTOTUNE = tf.data.experimental.AUTOTUNE
files_ds = tf.data.Dataset.zip((X_features_ds, Y_labels_ds))
waveform_ds = files_ds.map(get_wav_and_label, num_parallel_calls=AUTOTUNE)

shapes = tfds.as_numpy(waveform_ds.map(lambda x, _: tf.shape(x)))

max_padding_size = 0
for w in shapes:
    max_padding_size = max(max_padding_size, w[0])

def get_spectrogram(waveform):
    waveform = add_padding(waveform)
    spectrogram = tfio.experimental.audio.spectrogram(waveform, nfft=512, window=512, stride=256)
    # spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


def add_padding(waveform):
    waveform = tf.cast(waveform, tf.float32)
    zero_padding = tf.zeros([max_padding_size] - tf.shape(waveform), dtype=tf.float32)
    return tf.concat([waveform, zero_padding], 0)


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, [100, 100])
    return spectrogram, label


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    #spectrogram = np.resize((200, 100))
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


for waveform, label in waveform_ds.take(1):
    label = label.numpy()
    label = jobs_encoder.inverse_transform(np.array([label]))[0]
    spectrogram = get_spectrogram(waveform)

fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()

spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


def split_dataset(full_ds, train_size, test_size):
    ds_train = full_ds.take(train_size)
    ds_test = full_ds.skip(train_size)
    ds_val = ds_test.skip(test_size)
    ds_test = ds_test.take(test_size)

    return ds_train, ds_val, ds_test


BATCH_SIZE = 512

DATASET_SIZE = len(X_features)

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE
test_steps = test_size // BATCH_SIZE

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

ds_train, ds_val, ds_test = split_dataset(spectrogram_ds, train_size, test_size)

input_shape = next(iter(ds_train.take(1)))[0].shape

num_labels = len(jobs_encoder.classes_)

ds_train = ds_train.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)


model = models.Sequential([
    layers.Input(shape=input_shape),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(num_labels, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 20
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
)
model.save("data/model_with_padding.h5")
loss, acc = model.evaluate(ds_test, verbose=2)
print(loss)
print(acc)
logging.info(f"Loss: {loss}")
logging.info(f"Acc: {acc}")