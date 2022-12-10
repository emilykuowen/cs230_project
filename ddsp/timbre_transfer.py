# Ignore a bunch of deprecation warnings
import warnings
warnings.filterwarnings("ignore")

import copy
import os
import time

# print('Installing from pip package...')
# os.system("!pip install -qU ddsp==1.6.5 \"hmmlearn<=0.2.7\"")

import crepe
import ddsp
import ddsp.training
from ddsp.training.postprocessing import (
    detect_notes, fit_quantile_transform
)
import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from scipy.io import wavfile
import soundfile as sf

instrument = 'vocal'
# Audio should be monophonic (single instrument / voice)
# filename = 'test_audio/piano_low_Cmaj.wav'
filename = 'timbre_reference/vocal.wav'

audio, sample_rate = librosa.load(filename)
# print("audio shape: ", audio.shape)
if len(audio.shape) == 1:
  audio = audio[np.newaxis, :]
# print("new audio shape: ", audio.shape)

# Setup the session
ddsp.spectral_ops.reset_crepe()

# Compute features
start_time = time.time()
audio_features = ddsp.training.metrics.compute_audio_features(audio)
print("Audio features keys: ", audio_features.keys())
audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
print('Audio features took %.1f seconds' % (time.time() - start_time))

TRIM = -15
# Plot features
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
plt.savefig('input_audio_features.png')

model_dir = "model/" + instrument
# Load the dataset statistics
# DATASET_STATS = None
# dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
# print(f'Loading dataset statistics from {dataset_stats_file}')
# try:
#   if tf.io.gfile.exists(dataset_stats_file):
#     with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
#       DATASET_STATS = pickle.load(f)
#     #   print("Dataset statistics: ", DATASET_STATS)
# except Exception as err:
#   print('Loading dataset statistics from pickle failed: {}.'.format(err))

gin_file = os.path.join(model_dir, 'operative_config-0.gin')
# Parse gin config
with gin.unlock_config():
  gin.parse_config_file(gin_file, skip_unknown=True)

# Ensure dimensions and sampling rates are equal
time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
n_samples_train = gin.query_parameter('Harmonic.n_samples')
hop_size = int(n_samples_train / time_steps_train)

time_steps = int(audio.shape[1] / hop_size)
n_samples = time_steps * hop_size

print("===Trained model===")
print("Time Steps", time_steps_train)
print("Samples", n_samples_train)
print("Hop Size", hop_size)
print("\n===Resynthesis===")
print("Time Steps", time_steps)
print("Samples", n_samples)
print('')

gin_params = [
    'Harmonic.n_samples = {}'.format(n_samples),
    'FilteredNoise.n_samples = {}'.format(n_samples),
    'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
    'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
]

with gin.unlock_config():
  gin.parse_config(gin_params)

# Trim all input vectors to correct lengths 
for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
  audio_features[key] = audio_features[key][:time_steps]
audio_features['audio'] = audio_features['audio'][:, :n_samples]

# Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
ckpt_name = ckpt_files[0].split('.')[0]
print("using checkpoint ", ckpt_name)
ckpt = os.path.join(model_dir, ckpt_name)

# Set up the model just to predict audio given new conditioning
model = ddsp.training.models.Autoencoder()
model.restore(ckpt)

# Build model by running a batch through it.
start_time = time.time()
_ = model(audio_features, training=False)
print('Restoring model took %.1f seconds' % (time.time() - start_time))

# Run a batch of predictions.
start_time = time.time()
outputs = model(audio_features, training=False)
print("Model output keys: ", outputs.keys())
# for key in outputs.keys():
#     print(key, ": ", outputs[key])
#     print("")

harmonic_distribution = outputs['harmonic_distribution']
print("Harmonic distribution: ", harmonic_distribution)
print("Harmonic distribution shape: ", harmonic_distribution.shape)
harmonic_distribution = harmonic_distribution[:,:,0].numpy().flatten()
print("Trimmed harmonic distribution: ", harmonic_distribution)
print("Trimmed harmonic distribution shape: ", harmonic_distribution.shape)
np.save('harmonic_distribution_output/' + instrument + '_harmonic_distribution.npy', harmonic_distribution)

audio_gen = model.get_audio_from_outputs(outputs)
audio_gen = np.transpose(audio_gen.numpy())
sf.write('timbre_transfer_output/' + instrument + '.wav', audio_gen, sample_rate)

print('Prediction took %.1f seconds' % (time.time() - start_time))
