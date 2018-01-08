#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:32:53 2018

@author: truc
"""

import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from librosa.display import waveplot
plt.close('all')

sound_file_paths = "a039_10_20_forest_path.wav" #"a038_30_40_home.wav"

parent_dir = 'small_data/'

X,sr = librosa.load(os.path.join(parent_dir, sound_file_paths))

S = librosa.feature.melspectrogram(X, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(S, ref_power=np.max)

# Make a new figure
plt.figure()
librosa.display.waveplot(np.array(X),sr=22050,)

plt.figure()
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

plt.colorbar(format='%+02.0f dB')
plt.show()


from sklearn.decomposition import NMF
model = NMF(n_components=3, init='random', random_state=0)
W = model.fit_transform(S)
H = model.components_
print(W.shape)
print(H.shape)

model = NMF(n_components=3, init='random', random_state=0)
W1 = model.fit_transform(S.reshape(S.shape[1],S.shape[0]))
H1 = model.components_
print(W1.shape)
print(H1.shape)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(log_S)
kmeans.labels_


kmeans1 = KMeans(n_clusters=3, random_state=0).fit(log_S.reshape(S.shape[1],S.shape[0]))
kmeans1.labels_

dd = np.zeros(log_S.shape)
dd[np.where(kmeans.labels_==2)]=log_S[kmeans.labels_==2]
plt.figure()
librosa.display.specshow(dd, sr=sr, x_axis='time', y_axis='mel')

plt.figure()
res = kmeans.cluster_centers_[kmeans.labels_.flatten()]
res2 = res.reshape((log_S.shape))
plt.figure()
librosa.display.specshow(res2, sr=sr, x_axis='time', y_axis='mel')
plt.show
