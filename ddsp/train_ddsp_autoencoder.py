import glob
import os
import ddsp.training
import tensorboard as tb
import matplotlib.pyplot as plt
import numpy as np

# os.system("pip install -qU ddsp[data_preparation]==1.6.3")

instrument = 'piano'
"""
# Make directories to save model and data
DRIVE_DIR = '../../data/medleydb_' + instrument + '/' + instrument
AUDIO_DIR = '../../data/medleydb_' + instrument + '/tfaudio'
AUDIO_FILEPATTERN = AUDIO_DIR + '/*'
os.system("mkdir -p " + AUDIO_DIR)
SAVE_DIR = os.path.join(DRIVE_DIR, 'ddsp')
os.system("mkdir -p " + SAVE_DIR)

mp3_files = glob.glob(os.path.join(DRIVE_DIR, '*.mp3'))
wav_files = glob.glob(os.path.join(DRIVE_DIR, '*.wav'))
audio_files = mp3_files + wav_files

for fname in audio_files:
  target_name = os.path.join(AUDIO_DIR, os.path.basename(fname).replace(' ', '_'))
  print('Copying {} to {}'.format(fname, target_name))
  cp_command = "cp \"" + fname + "\" " + target_name
  os.system(cp_command)
"""

# Preprocess raw audio into TFRecord dataset
TRAIN_TFRECORD = '../../data/' + 'medleydb_' + instrument + '/tfrecord/train.tfrecord'
TRAIN_TFRECORD_FILEPATTERN = TRAIN_TFRECORD + '*'

"""
# # Copy dataset from drive if dataset has already been created.
# drive_data_dir = os.path.join(DRIVE_DIR, 'data')
# drive_dataset_files = glob.glob(drive_data_dir + '/*')

# if DRIVE_DIR and len(drive_dataset_files) > 0:
#     os.system("cp " + drive_data_dir + "/* data/" + instrument)
# else:

# Make a new dataset.
if not glob.glob(AUDIO_FILEPATTERN):
    raise ValueError('No audio files found. Please use the previous cell to upload.')

ddsp_prepare_tfrecord_command = "ddsp_prepare_tfrecord \
    --input_audio_filepatterns=" + AUDIO_FILEPATTERN + "\
    --output_tfrecord_path=" + TRAIN_TFRECORD + "\
    --num_shards=10 \
    --alsologtostderr"
os.system(ddsp_prepare_tfrecord_command)

# Copy dataset to drive for safe-keeping.
# if DRIVE_DIR:
#     os.system("mkdir " + drive_data_dir)
#     print('Saving to {}'.format(drive_data_dir))
#     os.system("cp " + TRAIN_TFRECORD_FILEPATTERN + " \"" + drive_data_dir + "\"/")
"""

data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
dataset = data_provider.get_dataset(shuffle=False)

try:
  ex = next(iter(dataset))
except StopIteration:
  raise ValueError(
      'TFRecord contains no examples. Please try re-running the pipeline with '
      'different audio file(s).')

f, ax = plt.subplots(3, 1, figsize=(14, 4))
x = np.linspace(0, 4.0, 1000)
ax[0].set_ylabel('loudness_db')
ax[0].plot(x, ex['loudness_db'])
ax[1].set_ylabel('F0_Hz')
ax[1].set_xlabel('seconds')
ax[1].plot(x, ex['f0_hz'])
ax[2].set_ylabel('F0_confidence')
ax[2].set_xlabel('seconds')
ax[2].plot(x, ex['f0_confidence'])
plt.savefig( 'tfaudio_features.png')

"""
Run the following command in terminal:
ddsp_prepare_tfrecord --input_audio_filepatterns="../../data/medleydb_piano/tfaudio/*" --output_tfrecord_path="../../data/medleydb_piano/tfrecord/train.tfrecord*" --num_shards=10 --alsologtostderr

ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="../data/train/bass/ddsp-solo-instrument" \
  --gin_file=models/solo_instrument.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='data/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --gin_param="train_util.train.num_steps=30000" \
  --gin_param="train_util.train.steps_per_save=300" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=10"

ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="models/piano" \
  --gin_file=models/solo_instrument.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='../../data/medleydb_piano/tfrecord/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --gin_param="train_util.train.num_steps=30000" \
  --gin_param="train_util.train.steps_per_save=300" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=10"
"""