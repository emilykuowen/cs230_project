# CS230 DDSP Music Source Separation Project
Group Members: Emily Kuo, Samantha Long, Sneha Shah

## Description
In an attempt to create a musical source separation model that can be conditioned to train different instruments, we have implemented a new framework built on an existing deep mask estimation model from the [ISMIR2020 tutorial](https://source-separation.github.io/tutorial/landing.html) with the use of a [DDSP Autoencoder](https://github.com/magenta/ddsp/blob/main/ddsp/training/models/autoencoder.py). We trained a DDSP model for each instrument we separated, and conditioned the network using harmonic distribution parameters generated by the trained DDSP model. The harmonic distribution context vector is passed into a fully connected layer introduced before the RNN. The fully connected layer modifies the activations using equations (1) and (2) below. We think this is exciting because the DDSP Autoencoder has been shown to be quite good at timbre transfer—converting the timbre of any sound to a specified instrument. This implies that the DDSP Autoencoder has the ability to learn the timbre features for an instrument well. The harmonic distribution parameter encodes important information about instrument timbre, which we hypothesize can help our model learn what an instrument sounds like.

Our current music source separation model can be trained to separate a single instrument. We have trained two models to separate bass and vocals respectively from a mixed signal. Putting it all together, we propose an architecture as shown in the figure below. Since the model uses an RNN without a U-net (as compared to \cite{ConditionedSS}), we apply the FiLM conditioning to a fully connected layer prior to passing input to the RNN model. Here, $\gamma$ and $\beta$ are trainable parameter vectors that control the parameters of the fully connected layer 1. They can be defined and related to the activation as follows:

$$\gamma, \beta = f(c)$$

$$\hat{a} = \gamma x + \beta$$

** INSERT PICTURE HERE **

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

Before training the model, you will need to make the following changes.

1. Run the following command in your terminal. If it cannot find the library, check the Requirements section to install the requirements.
```bash
python -m pip show nussl
```

2. Open the `separation_model.py` file using the following command, where * is the path displayed in the above step.
```bash
vi */nussl/
```

3. Make the following changes to the file
- Add the following line to the save function (line 243) 
```python
if 'condition' in self.config['modules']['model']['args'].keys():
  if isinstance(self.config['modules']['model']['args']['condition'], torch.Tensor):
    self.config['modules']['model']['args']['condition'] = self.config['modules']['model']['args']['condition'].tolist()
```
                
- Add `strict=False` to line 240 in the load function - 
```python
model.load_state_dict(model_dict['state_dict'], strict=False)
```
  
4. You're all set to train the source separation model!

The `ConditionedRecurrentStack` class in `nussl_modules.py` contains the model architecturethat can be run using one of the `*_train.py` codes.

Example command to train a bass separator model with conditioning:
```bash
# switch into the model directory
cd model
# run a model of your choice
python3 bass_separator_with_conditioning.py
```

## Useful Commands
To ssh into the AWS instance:
```bash
ssh -i "~/.ssh/cs230.pem" ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com
```

To run jupyter notebook on the AWS instance:
```bash
ssh -i "~/.ssh/cs230.pem" -fNL 9000:localhost:8888 ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com
```
