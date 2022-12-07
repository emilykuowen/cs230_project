# Ignore a bunch of deprecation warnings
import warnings
warnings.filterwarnings("ignore")

import copy
import os
import time

print('Installing from pip package...')
os.system("!pip install -qU ddsp==1.6.5 \"hmmlearn<=0.2.7\"")

import crepe
import ddsp
import ddsp.training
# from ddsp.colab.colab_utils import (
#     auto_tune, get_tuning_factor, download, 
#     play, record, specplot, upload, 
#     DEFAULT_SAMPLE_RATE)
from ddsp.training.postprocessing import (
    detect_notes, fit_quantile_transform
)
import gin
# from google.colab import files
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from scipy.io import wavfile

"""
Notes:
- Audio should be monophonic (single instrument / voice)
- Extracts fundmanetal frequency (f0) and loudness features. 
"""

filename = 'piano.wav'
sample_rate, audio = wavfile.read('piano.wav')

# Setup the session.
ddsp.spectral_ops.reset_crepe()

# Compute features.
start_time = time.time()
audio_features = ddsp.training.metrics.compute_audio_features(audio)
audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
audio_features_mod = None
print('Audio features took %.1f seconds' % (time.time() - start_time))

TRIM = -15
# Plot Features.
fig, ax = plt.subplots(nrows=3, 
                       ncols=1, 
                       sharex=True,
                       figsize=(6, 8))
ax[0].plot(audio_features['loudness_db'][:TRIM])
ax[0].set_ylabel('loudness_db')

ax[1].plot(librosa.hz_to_midi(audio_features['f0_hz'][:TRIM]))
ax[1].set_ylabel('f0 [midi]')

ax[2].plot(audio_features['f0_confidence'][:TRIM])
ax[2].set_ylabel('f0 confidence')
_ = ax[2].set_xlabel('Time step [frame]')
plt.savefig('piano_audio_features.png')
