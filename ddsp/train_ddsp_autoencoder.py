import glob
import os
import ddsp.training
from ddsp.colab import colab_utils

os.system("pip install -qU ddsp[data_preparation]==1.6.3")

DRIVE_DIR = '../data/train/'
AUDIO_DIR = 'data/audio'
AUDIO_FILEPATTERN = AUDIO_DIR + '/*'
os.system("mkdir -p " + AUDIO_DIR)
SAVE_DIR = os.path.join(DRIVE_DIR, 'ddsp-solo-instrument')
os.system("mkdir -p " + SAVE_DIR)
mp3_files = glob.glob(os.path.join(DRIVE_DIR, '*.mp3'))
wav_files = glob.glob(os.path.join(DRIVE_DIR, '*.wav'))
audio_files = mp3_files + wav_files

for fname in audio_files:
  target_name = os.path.join(AUDIO_DIR, os.path.basename(fname).replace(' ', '_'))
  print('Copying {} to {}'.format(fname, target_name))
  os.system("cp " + fname + " " + target_name)

TRAIN_TFRECORD = 'data/train.tfrecord'
TRAIN_TFRECORD_FILEPATTERN = TRAIN_TFRECORD + '*'

# Copy dataset from drive if dataset has already been created.
drive_data_dir = os.path.join(DRIVE_DIR, 'data')
drive_dataset_files = glob.glob(drive_data_dir + '/*')

if DRIVE_DIR and len(drive_dataset_files) > 0:
    os.system("cp " + drive_data_dir + "/* data/")

else:
    # Make a new dataset.
    if not glob.glob(AUDIO_FILEPATTERN):
        raise ValueError('No audio files found. Please use the previous cell to upload.')

    ddsp_command = "ddsp_prepare_tfrecord \
        --input_audio_filepatterns=" + AUDIO_FILEPATTERN + "\
        --output_tfrecord_path=" + TRAIN_TFRECORD + "\
        --num_shards=10 \
        --alsologtostderr"
    os.system(ddsp_command)
    # Copy dataset to drive for safe-keeping.
    if DRIVE_DIR:
        os.system("mkdir " + drive_data_dir)
        print('Saving to {}'.format(drive_data_dir))
        os.system("cp " + TRAIN_TFRECORD_FILEPATTERN + "$drive_data_dir")

data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
dataset = data_provider.get_dataset(shuffle=False)
PICKLE_FILE_PATH = os.path.join(SAVE_DIR, 'dataset_statistics.pkl')

_ = colab_utils.save_dataset_statistics(data_provider, PICKLE_FILE_PATH, batch_size=1)