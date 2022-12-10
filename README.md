# CS230 DDSP Music Source Separation Project
Group Members: Emily Kuo, Samantha Long, Sneha Shah

## Requirements
```bash
# To install the requirements needed for our code
pip install -r requirements.txt
# To install ffmpeg in a conda environment
conda install -c conda-forge ffmpeg 
```

## Usage
# Data

# DDSP Pre-training

# Deep Mask Estimation Training

## Useful Commands
```bash
# To ssh into the AWS instance:
ssh -i "~/.ssh/cs230.pem" ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com

# To run jupyter notebook on the AWS instance:
ssh -i "~/.ssh/cs230.pem" -fNL 9000:localhost:8888 ubuntu@ec2-54-149-20-20.us-west-2.compute.amazonaws.com
```