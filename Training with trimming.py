import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import logging
import tensorflow_hub as hub
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow_transform import scale_by_min_max
import tensorflow_io as tfio
from tools import get_wav
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def make_path(path, x):
    with_part = path.joinpath(x.part)
    return str(Path(with_part, x.path.replace("mp3", "wav")))


logging.basicConfig(filename='training_without_padding_50.log', level=logging.INFO)

df = pd.read_csv(r"I:\accent_300K\df_accent.csv")
df = df.sample(n=1000).reset_index(drop=True)

# unique = df["accent"].unique()
path = Path(r"I:\accent_300K\cv-corpus-6.1-2020-12-11\en")
df["path"] = df.apply(lambda x: make_path(path, x), axis=1)
df = df.sample(frac=1)

jobs_encoder = LabelBinarizer()
jobs_encoder.fit(df['accent'])
transformed = jobs_encoder.transform(df['accent'])
ohe_df = pd.DataFrame(transformed)
classes = jobs_encoder.classes_.tolist()
X_features = df["path"].to_numpy()
Y_labels = ohe_df.to_numpy()

X_train, X_testval, Y_train, Y_testval = train_test_split(X_features, Y_labels, train_size=0.7, test_size=0.30)
X_test, X_val, Y_test, Y_val = train_test_split(X_testval,
                                                Y_testval,
                                                train_size=0.5,
                                                test_size=0.50)

AUTOTUNE = tf.data.experimental.AUTOTUNE
print("Finished Setting up paths")

X_all = tf.convert_to_tensor(X_features, dtype=tf.string)
Y_all = tf.convert_to_tensor(Y_labels, dtype=tf.float32)

X_train = tf.convert_to_tensor(X_train, dtype=tf.string)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

X_test = tf.convert_to_tensor(X_test, dtype=tf.string)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

X_val = tf.convert_to_tensor(X_val, dtype=tf.string)
Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float32)

X_train_ds = tf.data.Dataset.from_tensor_slices(X_train)

Y_train_ds = tf.data.Dataset.from_tensor_slices(Y_train)

X_test_ds = tf.data.Dataset.from_tensor_slices(X_test)
Y_test_ds = tf.data.Dataset.from_tensor_slices(Y_test)

X_val_ds = tf.data.Dataset.from_tensor_slices(X_val)
Y_val_ds = tf.data.Dataset.from_tensor_slices(Y_val)

X_all_ds = tf.data.Dataset.from_tensor_slices(X_all)
Y_all_ds = tf.data.Dataset.from_tensor_slices(Y_all)

files_train_ds = tf.data.Dataset.zip((X_train_ds, Y_train_ds))
files_test_ds = tf.data.Dataset.zip((X_test_ds, Y_test_ds))
files_val_ds = tf.data.Dataset.zip((X_val_ds, Y_val_ds))
all_ds = tf.data.Dataset.zip((X_all_ds, Y_all_ds))

BATCH_SIZE = 64
print("Finished Setting up datasets")


RATE = 16000
frame_length = 1024
step_time = 0.008
frame_step = int(RATE * step_time)


def preprocess_feature_label(file_name, label):
    wav = get_wav(file_name)

    position = tfio.audio.trim(wav, axis=0, epsilon=0.1)
    start = position[0]
    stop = position[1]

    wav = wav[start:stop]

    wav = wav[0:16000]

    wav = tf.cast(wav, tf.float32)
    zero_padding = tf.zeros([RATE] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([wav, zero_padding], 0)

    spectrogram = tfio.audio.spectrogram(wav, nfft=512, window=512, stride=frame_step)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = scale_by_min_max(spectrogram)
    spectrogram = tf.image.resize(spectrogram, (100, 100))
    return spectrogram, label


spectrogram_train_ds = files_train_ds.map(preprocess_feature_label, num_parallel_calls=AUTOTUNE)
spectrogram_test_ds = files_test_ds.map(preprocess_feature_label, num_parallel_calls=AUTOTUNE)
spectrogram_val_ds = files_val_ds.map(preprocess_feature_label, num_parallel_calls=AUTOTUNE)

DATASET_SIZE = len(X_features)

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE
test_steps = test_size // BATCH_SIZE

input_shape = next(iter(spectrogram_train_ds.take(1)))[0].shape

num_labels = len(jobs_encoder.classes_)

logging.info("Final Touches")
ds_train = spectrogram_train_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
ds_val = spectrogram_val_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

print("Finished Everything")
model = models.Sequential([
    layers.Input(shape=input_shape, dtype=tf.float32, name='audio'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(num_labels, activation="softmax"),
])
print("Finished Compiling")
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
)
plot_model(model, to_file="./figures/model2.png", show_shapes=True, show_layer_names=True)
EPOCHS = 30
logging.info("Starting Training")
print("Starting Training")
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
)

model.save("data/model_without_padding_50.h5")

ds_test = spectrogram_test_ds

test_audio = []
test_labels = []

for audio, label in ds_test:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = np.argmax(test_labels, axis=1)

acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {acc:.0%}')

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
sns.heatmap(confusion_mtx, xticklabels=classes, yticklabels=classes,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.tight_layout()
plt.savefig("./figures/CM_50K_trim_only.png")

print(acc)
logging.info(f"Acc: {acc}")
plt.show()