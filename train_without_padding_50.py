import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_io as tfio
import logging


def make_path(path, x):
    with_part = path.joinpath(x.part)
    return str(Path(with_part, x.path.replace("mp3", "wav")))


# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

logging.basicConfig(filename='training_without_padding_50.log', level=logging.INFO)

df = pd.read_csv(r"I:\accent_300K\df_accent.csv")
df = df.sample(frac=1).reset_index(drop=True)

# unique = df["accent"].unique()
path = Path(r"I:\accent_300K\cv-corpus-6.1-2020-12-11\en")
df["path"] = df.apply(lambda x: make_path(path, x), axis=1)
df = df.sample(frac=1)

jobs_encoder = LabelBinarizer()
jobs_encoder.fit(df['accent'])
transformed = jobs_encoder.transform(df['accent'])
ohe_df = pd.DataFrame(transformed)

X_features = df["path"].to_numpy()
Y_labels = ohe_df.to_numpy()

X_train, X_testval, Y_train, Y_testval = train_test_split(X_features, Y_labels, train_size=0.7, test_size=0.30)
X_test, X_val, Y_test, Y_val = train_test_split(X_testval
                                                , Y_testval, train_size=0.5, test_size=0.50)

AUTOTUNE = tf.data.experimental.AUTOTUNE
print("Finished Setting up paths")


def decode_audio(audio_binary):

    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    return audio


def get_wav_and_label(file_name: tf.string, label):
    audio_binary = tf.io.read_file(file_name)
    wav = decode_audio(audio_binary)
    return wav, label


X_all = tf.convert_to_tensor(X_features, dtype=tf.string)
Y_all = tf.convert_to_tensor(Y_labels, dtype=tf.float32)

X_train = tf.convert_to_tensor(X_train, dtype=tf.string)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

X_test = tf.convert_to_tensor(X_test, dtype=tf.string)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

X_val = tf.convert_to_tensor(X_val, dtype=tf.string)
Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float32)

X_all_ds = tf.data.Dataset.from_tensor_slices(X_all)
Y_all_ds = tf.data.Dataset.from_tensor_slices(Y_all)

X_train_ds = tf.data.Dataset.from_tensor_slices(X_train)
Y_train_ds = tf.data.Dataset.from_tensor_slices(Y_train)

X_test_ds = tf.data.Dataset.from_tensor_slices(X_test)
Y_test_ds = tf.data.Dataset.from_tensor_slices(Y_test)

X_val_ds = tf.data.Dataset.from_tensor_slices(X_val)
Y_val_ds = tf.data.Dataset.from_tensor_slices(Y_val)

files_train_ds = tf.data.Dataset.zip((X_train_ds, Y_train_ds))
files_test_ds = tf.data.Dataset.zip((X_test_ds, Y_test_ds))
files_val_ds = tf.data.Dataset.zip((X_val_ds, Y_val_ds))

files_all_ds = tf.data.Dataset.zip((X_all_ds, Y_all_ds))


BATCH_SIZE = 512
waveform_all_ds = files_all_ds.map(get_wav_and_label, num_parallel_calls=AUTOTUNE)
waveform_train_ds = files_train_ds.map(get_wav_and_label, num_parallel_calls=AUTOTUNE)
waveform_test_ds = files_test_ds.map(get_wav_and_label, num_parallel_calls=AUTOTUNE)
waveform_val_ds = files_val_ds.map(get_wav_and_label, num_parallel_calls=AUTOTUNE)
print("Finished Setting up datasets")


def get_spectrogram(waveform):
    spectrogram = tfio.audio.spectrogram(waveform, nfft=512, window=512, stride=256)
    # spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, [100, 100])
    return spectrogram, label


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


spectrogram_train_ds = waveform_train_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
spectrogram_test_ds = waveform_test_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
spectrogram_val_ds = waveform_val_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


def split_dataset(full_ds, train_size, test_size):
    ds_train = full_ds.take(train_size)
    ds_test = full_ds.skip(train_size)
    ds_val = ds_test.skip(test_size)
    ds_test = ds_test.take(test_size)

    return ds_train, ds_val, ds_test


DATASET_SIZE = len(X_features)

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE
test_steps = test_size // BATCH_SIZE

logging.info("Normalization")
norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_train_ds.map(lambda x, _: x, num_parallel_calls=AUTOTUNE))

input_shape = next(iter(spectrogram_train_ds.take(1)))[0].shape

num_labels = len(jobs_encoder.classes_)

logging.info("Final Touches")
ds_train = spectrogram_train_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
ds_val = spectrogram_val_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)


print("Finished Everything")
model = models.Sequential([
    layers.Input(shape=input_shape),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(num_labels, activation="softmax"),
])
print("Finished Compiling")
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
)

EPOCHS = 20
logging.info("Starting Training")
print("Starting Training")
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    verbose=2,
)

model.save("data/model_without_padding_50.h5")
ds_test = spectrogram_test_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
loss, acc = model.evaluate(ds_test, verbose=2)
print(loss)
print(acc)
logging.info(f"Loss: {loss}")
logging.info(f"Acc: {acc}")
