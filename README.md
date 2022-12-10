# CS230 DDSP Music Source Separation Project
Group Members: Emily Kuo, Samantha Long, Sneha Shah

## Description
Describe our project

## Requirements
To install the requirements needed for our code:
```bash
pip install -r requirements.txt
```
To install ffmpeg in a conda environment:
```bash
conda install -c conda-forge ffmpeg 
```

## Usage
### Data
We mainly use two existing datasets, [MUSDB18](https://zenodo.org/record/1117372#.Y5Pfv-zMLdo) and [MedleyDB](https://medleydb.weebly.com/), to train our music source separation model.
- MUSDB18 is downloaded by the library [nussl](https://github.com/nussl/nussl).
- The preprocessed version of MedleyDB can be downloaded by the following commands (be aware of the large file sizes):
```bash
# To download the acoustic guitar stems (4.3 GB)
aws s3 sync s3://medleydb/acoustic_guitar/ "path to your data folder"
# To download the bass stems (9.94 GB)
aws s3 sync s3://medleydb/bass/ "path to your data folder"
# To download the flute stems (2.63 GB)
aws s3 sync s3://medleydb/flute/ "path to your data folder"
# To download the piano stems (11.51 GB)
aws s3 sync s3://medleydb/piano/ "path to your data folder"
# To download the violin stems (5.27 GB)
aws s3 sync s3://medleydb/violin/ "path to your data folder"
```

### DDSP Pre-training
We used the [DDSP](https://github.com/magenta/ddsp) model to produce the conditioning input for our model. We define the conditioning input as the harmonic distribution of the instrument we want to separate. For example, to generate the conditioning input of a bass separator, we would run a single-note bass audio file through a DDSP model pretrained on bass sounds to get the harmonic distrbution of bass.

We've previously trained DDSP models and generated conditioning inputs for bass and vocals. Those conditioning inputs are stored as `.npy` files in the folder `ddsp/harmonic_distribution_output/`.

To train a DDSP model on a new instrument using your own data, follow these steps:
1. In `train_ddsp_autoencoder.py`, change the string `instrument` to the instrument you'd like to train the model on (this is so that the output folder paths get changed correspondingly).
2. Switch into the `ddsp/` folder:
```bash
cd ddsp
```
3. Generate the tfrecord files that correspond to your own data:
```bash
python3 train_ddsp_autoencoder.py
```
The tfrecord files will be stored in `ddsp/data/"instrument_name"`.
4. Run DDSP training
```bash
ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="path to the folder where you want to store the checkpoints" \
  --gin_file=model/solo_instrument.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='path to the folder where train.tfrecord* are stored'" \
  --gin_param="batch_size=16" \
  --gin_param="train_util.train.num_steps=30000" \
  --gin_param="train_util.train.steps_per_save=300" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=5"
```
Example command:
```bash
ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="model/vocals_medleydb" \
  --gin_file=model/solo_instrument.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='data/vocals_medleydb/train.tfrecord*'" \
  --gin_param="batch_size=16" \
  --gin_param="train_util.train.num_steps=30000" \
  --gin_param="train_util.train.steps_per_save=300" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=5"
```

### Model Training

## Useful Commands
To ssh into the AWS instance:
```bash
ssh -i "~/.ssh/cs230.pem" ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com
```

To run jupyter notebook on the AWS instance:
```bash
ssh -i "~/.ssh/cs230.pem" -fNL 9000:localhost:8888 ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com
```