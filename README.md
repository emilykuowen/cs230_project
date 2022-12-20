# CS230 Project - Music Source Separation with DDSP
Group Members: Emily Kuo, Samantha Long, Sneha Shah


## Description
In an attempt to create a music source separation model that can be conditioned to train different instruments, we have implemented a new framework built on an existing deep mask estimation model from the [ISMIR2020 tutorial](https://source-separation.github.io/tutorial/landing.html) with the use of a [DDSP Autoencoder](https://github.com/magenta/ddsp/blob/main/ddsp/training/models/autoencoder.py). We trained a DDSP Autoencoder for each instrument we separated and conditioned the mask estimation model using harmonic distribution parameters generated by the trained DDSP Autoencoder. The harmonic distribution context vector is passed into a fully connected layer introduced before the RNN. The fully connected layer modifies the activations using the equations below. We think this is exciting because the DDSP Autoencoder has been shown to be good at timbre transfer (converting the timbre of any sound to a specified instrument). This implies that the DDSP Autoencoder has the ability to learn the timbre features of an instrument well. The harmonic distribution parameters encode important information about instrument timbre, which we hypothesize can help our model learn what an instrument sounds like.

Our current music source separation model can be trained to separate a single instrument. We have trained models to separate bass, vocals, and drums from a music mix. Putting it all together, we propose an architecture as shown in the figure below. Since the model uses an RNN without a U-net (as compared to the paper [here](https://arxiv.org/abs/2004.03873)), we apply FiLM conditioning to a fully connected layer prior to passing input to the RNN model. Here, $\gamma$ and $\beta$ are trainable parameter vectors that control the parameters of the fully connected layer 1. They can be defined and related to the activation as follows:

$$\gamma, \beta = f(c)$$

$$\hat{a} = \gamma x + \beta$$

<p align="center">
<img src="https://user-images.githubusercontent.com/54175817/206850778-eb98dd9f-745f-48db-bf28-387e41fc9e39.jpg" alt="novel_arch_fin" style="max-width: 100%;" width="300">
</p>


## Requirements
To install the requirements needed for our code:
```bash
pip install -r requirements.txt
```
To install ffmpeg in a conda environment:
```bash
conda install -c conda-forge ffmpeg 
```


## Data
We mainly use two existing datasets, [MUSDB18](https://zenodo.org/record/1117372#.Y5Pfv-zMLdo) and [MedleyDB](https://medleydb.weebly.com/), to train our music source separation model.
### MUSDB18
[MUSDB18](https://zenodo.org/record/1117372#.Y5Pfv-zMLdo) includes 10 hours of 150 full length musical tracks of different genres along with their isolated drums, bass, vocals, and other categories of stems.
- MUSDB18 is downloaded by the libraries [common](https://github.com/source-separation/tutorial/tree/master/common) and [nussl](https://github.com/nussl/nussl). In the main function of model files like `model/bass_separator.py`, the function `data.prepare_musdb(dataset_path)` downloads 7-second segments of the MUSDB18 dataset and the function `nussl.datasets.MUSDB18(subsets=['test'], transform=tfm)` allows you to access a specific subset.

### MedleyDB
[MedleyDB](https://medleydb.weebly.com/) 1.0 and 2.0 contain 196 total multi-tracks with mixed and processed stems along with raw audio with annotations and metadata. Unlike MUSDB18 that has fixed instrument stems for each track, MedleyDB has different instrument stems for each track. Thus, in order to extract tracks that contain the instrument we'd like to separate, we wrote `medley_proprocess.py`, which you can run to extract instrument stems from the dataset.

We've processed the MedleyDB dataset to extract the following instruments: acoustic guitar, bass, flute, piano, and violin. If you'd like to download them, you can run the commands below (beware of the large file sizes):
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

If you'd like to extract stems of other instruments in MedleyDB, you can follow these steps:
1. Download the full MedleyDB 1.0 dataset [here](https://zenodo.org/record/1649325#.Y5UADuzMJ4l) and the MedleyDB 2.0 dataset [here](https://zenodo.org/record/1715175#.Y5UAD-zMJ4l).
2. Download the [metadata folder](https://github.com/marl/medleydb/tree/master/medleydb/data/Metadata) that contains stem information about each track.
3. Make the following changes in `data/medley_preprocess.py`
- Edit `track_dir` to the folder path that contains all of your MedleyDB track folders.
- Edit `metadata_dir` to the folder path that contains the MedleyDB metadata.
4. Run `python data/medley_preprocess.py`


## DDSP Training
We used the [DDSP](https://github.com/magenta/ddsp) model to produce the conditioning input for our model. We define the conditioning input as the harmonic distribution of the instrument we want to separate. For example, to generate the conditioning input of a bass separator, we would run a single-note bass audio file through a DDSP model pretrained on bass sounds to get the harmonic distrbution of bass.

We've previously trained DDSP models and generated conditioning inputs for bass, vocals, and drums. Those conditioning inputs are stored as `.npy` files in the folder `ddsp/harmonic_distribution_output/`.

To train a DDSP model on a new instrument using your own data, follow these steps:
1. In `ddsp/generate_ddsp_tfrecord.py`, change the string `instrument` to the instrument you'd like to train the model on (this is so that the output folder paths get changed accordingly).
2. Switch into the `ddsp/` folder:
```bash
cd ddsp
```
3. Generate the tfrecord files that correspond to your own data. The tfrecord files will be stored in `ddsp/data/"instrument"`.
```bash
python3 generate_ddsp_tfrecord.py
```
4. Run DDSP training by the following terminal command
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

Then, to get the harmonic distribution output from a DDSP model, follow these steps:
1. In `ddsp/timbre_transfer.py`, make the following changes:
- Edit `instrument` to the name of the instrument you'd like to train on
- Edit `filename` to the file path of the audio file you'd like to pass through the model
2. Run the following command in the ddsp folder:
```bash
python3 timbre_transfer.py
```
3. Check the .wav files in `ddsp/timbre_transfer_output/` folder to hear how good your DDSP model performs.
4. The harmonic distribution features will be saved as a `.npy` file in the folder `ddsp/harmonic_distribution_output/`, which you can later retrieve by running the function `np.load(filepath)`.


## Source Separation Model Training
All the model training code are stored in the `model/` folder. 

Before training a model, you will need to make the following changes in your local `nussl` library:
1. Run the following command in your terminal. If it cannot find `nussl`, check above to install the requirements.
```bash
python -m pip show nussl
```
2. Open the `separation_model.py` file using the following command, where * is the path displayed in the above step.
```bash
vi */nussl/ml/networks/separation_model.py 
```
3. Make the following changes to the file
- Add the following line to the save function (line 243) 
```python
if 'condition' in self.config['modules']['model']['args'].keys():
  if isinstance(self.config['modules']['model']['args']['condition'], torch.Tensor):
    self.config['modules']['model']['args']['condition'] = self.config['modules']['model']['args']['condition'].tolist()
```             
- Add `strict=False` in the load function (line 240)
```python
model.load_state_dict(model_dict['state_dict'], strict=False)
```
4. You're all set to train the source separation model!

### Example
To train a bass separator model with conditioning:
1. Specify hyperparameters (`batch_size`, `max_epochs`, and `epoch_length`) in the `train()` function on line 267.
2. Run the following terminal commands
```bash
# switch into the model directory
cd model
python3 bass_separator_with_conditioning.py
```
- This would create a subfolder `bass_output_with_conditioning/`, which contains all of the model checkpoints, audio output files, and .json files that record the model evaluation results. All of the `nussl` evaluation metrics will be printed in the terminal as well.


## Useful Commands
To ssh into the AWS instance:
```bash
ssh -i "~/.ssh/cs230.pem" ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com
```

To run jupyter notebook on the AWS instance:
```bash
ssh -i "~/.ssh/cs230.pem" -fNL 9000:localhost:8888 ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com
```
