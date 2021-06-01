import numpy
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
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
print(helpers.get_info(df, "accent"))

#%%
df = df.sample(frac=1).reset_index(drop=True)

#%%
def make_path(path, x):
    with_part = path.joinpath(x.part)
    return str(Path(with_part, x.path.replace("mp3", "wav")))

path = Path(r"C:\Users\Magnus\Desktop\DM\projekt\Datamining---Audio-\data\data_small\cv-corpus-6.1-2020-12-11\en")
df["path"] = df.apply(lambda x: make_path(path, x), axis=1)
df = df.sample(frac=1)

#%%
'''
    Maps labels to one hot arrays
'''

def encoding1():
    jobs_encoder = LabelBinarizer()
    jobs_encoder.fit(df['accent'])
    transformed = jobs_encoder.transform(df['accent'])
    ohe_df = pd.DataFrame(transformed)
    return ohe_df.to_numpy()

'''
    Maps labels to [number]
'''

def encoding2():
    lb = LabelEncoder()
    trans = lb.fit_transform(df["accent"])
    encoded = np.array(list(map(lambda x: [float(x)], trans)))
    print(lb.classes_)
    return encoded


'''
    Maps labels to number
'''

def encoding3():
    lb = LabelEncoder()
    vector = np.vectorize(np.float32)
    trans = lb.fit_transform(df["accent"])
    return vector(trans)


def encoding4():
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
    y = df[["accent"]]
    y["accent"] = y["accent"].map(label_mapping)
    y = np.array(y)
    return y

#%%
ohe = encoding1()
print(ohe)


#%%
X_features = df["path"].apply(audiofunctions.single_freq_hist).to_numpy()
Y_labels = ohe

#%%
print(X_features)

#%%

X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_labels, test_size=0.2)
#X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, train_size=0.7, test_size=0.15)


#%%
print(Y_train.shape)
print(X_train.shape)

#%%


#%%
for ele in X_train:
    print(len(ele))

print(len(X_train))

#%%
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform")
knn.fit(X_train, Y_train)

prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, prediction)
print("Predictions: ", prediction)
print("Accuracy: ", accuracy)
