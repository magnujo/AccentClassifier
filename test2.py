import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from tensorflow.keras import backend as K

df = pd.read_csv(r"D:\data_small\df_accent_small.csv")

# unique = df["accent"].unique()
path = Path(r"D:\data_small\cv-corpus-6.1-2020-12-11\en\wav")
df["path"] = df["path"].apply(lambda x: str(path.joinpath(x.replace("mp3", "wav"))))
df = df.sample(frac=1)

jobs_encoder = LabelBinarizer()
jobs_encoder.fit(df['accent'])
transformed = jobs_encoder.transform(df['accent'])
ohe_df = pd.DataFrame(transformed)

X_features = df["path"].to_numpy()
Y_labels = ohe_df.to_numpy()

maxData = 30000
model_name = "model_age"
frame_length = 1024
spect_length = int(frame_length / 2 + 1)
step_time = 0.008
image_width = 100  # 100*0.008=800ms
classesCount = len(jobs_encoder.classes_)
batch_size = 64
epochs = 15


def audioToTensor(filepath: str):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    frame_step = int(audioSR * step_time)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spect_real = tf.math.real(spectrogram)
    spect_real = tf.abs(spect_real)
    partsCount = len(spectrogram) // image_width
    parts = np.zeros((partsCount, image_width, spect_length))
    for i, p in enumerate(range(0, len(spectrogram) - image_width, image_width)):
        parts[i] = spect_real[p:p + image_width]
    return parts, audioSR


parts_count = 0
min_parts = 100
for i in range(0, len(Y_labels)):
    parts1 = len(audioToTensor(X_features[i]))
    parts_count += parts1
    min_parts = min(min_parts, parts1)

print("parts_count:", parts_count)
print("min_parts:", min_parts)


class MySequence(tf.keras.utils.Sequence):
    def __init__(self, x_voice, y_accent, batch_size, parts_count, min_parts):
        self.x_voice, self.y_accent = x_voice, y_accent
        self.batch_size = batch_size
        self.parts_count = parts_count
        self.min_parts = min_parts

    def __len__(self):
        return (len(self.x_voice) * self.min_parts) // self.batch_size

    def __getitem__(self, idx):
        batch_x_voice = np.zeros((batch_size, image_width, int(frame_length / 2 + 1)))
        batch_y_accent = np.zeros((batch_size, classesCount))
        for i in range(0, batch_size // self.min_parts):
            # print(idx, self.batch_size, self.min_parts, i, idx * self.batch_size//self.min_parts + i)
            age = self.y_accent[idx * self.batch_size // self.min_parts + i]
            voice, _ = audioToTensor(self.x_voice[idx * self.batch_size // self.min_parts + i])
            for j in range(0, self.min_parts):
                batch_x_voice[i * self.min_parts + j] = random.choice(voice)
                batch_y_accent[i * self.min_parts + j] = age
        return batch_x_voice, batch_y_accent


def accent_mae(y_true, y_pred):
    true_accent = K.sum(y_true * K.arange(15, classesCount * 10 + 10, 10, dtype="float32"), axis=-1)
    pred_accent = K.sum(y_pred * K.arange(15, classesCount * 10 + 10, 10, dtype="float32"), axis=-1)
    return K.mean(K.abs(true_accent - pred_accent))


main_input = Input(shape=(image_width, spect_length), name='main_input')
x = main_input
x = Reshape((image_width, spect_length, 1))(x)
x = preprocessing.Resizing(image_width // 2, spect_length // 2)(x)
x = Conv2D(34, 3, activation='relu')(x)
x = Conv2D(64, 3, activation='relu')(x)
x = MaxPooling2D()(x)
x = Dropout(0.1)(x)
x = Flatten()(x)
x = Dense(classesCount, activation='softmax')(x)
model = Model(inputs=main_input, outputs=x)
#tf.keras.utils.plot_model(model, to_file=model_name + '.png', show_shapes=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[accent_mae])


history = model.fit(MySequence(X_features[:int(len(X_features)*0.8)], Y_labels[:int(len(X_features)*0.8)],
                               batch_size, parts_count, min_parts),
                    epochs=epochs,
                    validation_data=MySequence(X_features[int(len(X_features)*0.8):],
                                               Y_labels[int(len(Y_labels)*0.8):], batch_size, parts_count, min_parts))
