#%%
import numpy
from sklearn import neighbors, metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, scale, StandardScaler
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_io as tfio
import logging
import helpers
import make_parts
import audiofunctions


# higher k if dataset is big. k should be odd

#%%
df = pd.read_csv(r"C:\Users\Magnus\Desktop\DM\projekt\Datamining---Audio-\data\data_small\df_accent.csv")

df = df.sample(frac=1).reset_index(drop=True)

#%%
def make_path(path, x):
    with_part = path.joinpath(x.part)
    return str(Path(with_part, x.path.replace("mp3", "wav")))

path = Path(r"C:\Users\Magnus\Desktop\DM\projekt\Datamining---Audio-\data\data_small\cv-corpus-6.1-2020-12-11\en")
df["path"] = df.apply(lambda x: make_path(path, x), axis=1)
df = df.sample(frac=1)

#%%
df = df[["path", "accent", "part"]]


#%%
X = []
for ele in df["path"].values:
    t = audiofunctions.single_freq_hist(ele)
    X.append(t)

X = np.array(X)


#%%
X = StandardScaler().fit_transform(X)

#%%
X = PCA().fit_transform(X)

#%%
y = df["accent"]

label_mapping = {
        'african': 0,
        'australia': 1,
        'bermuda': 2,
        'canada': 3,
        'england': 4,
        'hongkong': 5,
        'indian': 6,
        'ireland': 7,
        'malaysia': 8,
        'philippines': 9,
        'scotland': 10,
        'us': 11
    }
y = y.map(label_mapping)
y = np.array(y)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%
knn = neighbors.KNeighborsClassifier(n_neighbors=9, weights="uniform")

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)

print(prediction)
print(accuracy)

