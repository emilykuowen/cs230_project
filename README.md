# CS230 DDSP Music Source Separation Project
Group Members: Emily Kuo, Samantha Long, Sneha Shah

## Requirements
To install the requirements needed for our code
```bash
pip install -r requirements.txt
```
To install ffmpeg in a conda environment:
```bash
conda install -c conda-forge ffmpeg 
```

## Usage
### Data
We mainly use two existing datasets, [MUSDB18](https://zenodo.org/record/1117372#.Y5Pfv-zMLdo) and [MedleyDB](https://medleydb.weebly.com/), to train our model.
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

### Deep Mask Estimation Training

## Useful Commands
```bash
# To ssh into the AWS instance:
ssh -i "~/.ssh/cs230.pem" ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com

# To run jupyter notebook on the AWS instance:
ssh -i "~/.ssh/cs230.pem" -fNL 9000:localhost:8888 ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com
```