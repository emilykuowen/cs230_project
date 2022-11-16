import os
from common import data
import nussl
import torch
import tqdm
from nussl.datasets import transforms as nussl_tfm
from common.models import MaskInference
from common import utils, data
from pathlib import Path
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from torch import nn
import matplotlib.pyplot as plt
import json
import glob
import numpy as np


class MaskInference(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources, 
                activation='sigmoid'):
        super().__init__()
        
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features * num_audio_channels, hidden_size, 
            num_layers, bool(bidirectional), dropout
        )
        hidden_size = hidden_size * (int(bidirectional) + 1)
        self.embedding = Embedding(num_features, hidden_size, 
                                   num_sources, activation, 
                                   num_audio_channels)
        
    def forward(self, data):
        mix_magnitude = data # save for masking
        
        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        estimates = mix_magnitude.unsqueeze(-1) * mask
        
        output = {
            'mask': mask,
            'estimates': estimates
        }
        return output
    
    # Added function
    @classmethod
    def build(cls, num_features, num_audio_channels, hidden_size, 
              num_layers, bidirectional, dropout, num_sources, 
              activation='sigmoid'):
        # Step 1. Register our model with nussl
        nussl.ml.register_module(cls)
        
        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'MaskInference',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'activation': activation
                }
            }
        }
        
        # Step 2b: Define the connections between input and output.
        # Here, the mix_magnitude key is the only input to the model.
        connections = [
            ['model', ['mix_magnitude']]
        ]
        
        # Step 2c. The model outputs a dictionary, which SeparationModel will
        # change the keys to model:mask, model:estimates. The lines below 
        # alias model:mask to just mask, and model:estimates to estimates.
        # This will be important later when we actually deploy our model.
        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            # Bug fix: added brackets around f'model:{key}' to avoid KeyError
            connections.append([key, [f'model:{key}']])
        
        # Step 2d. There are two outputs from our SeparationModel: estimates and mask.
        # Then put it all together.
        output = ['estimates', 'mask',]
        config = {
            'name': cls.__name__,
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)


def train_step(engine, batch):
    optimizer.zero_grad()
    output = model(batch) # forward pass
    loss = loss_fn(
        output['estimates'],
        batch['source_magnitudes']
    )
    
    loss.backward() # backwards + gradient step
    optimizer.step()
    
    loss_vals = {
        'L1Loss': loss.item(),
        'loss': loss.item()
    } 
    
    return loss_vals


def val_step(engine, batch):
    with torch.no_grad():
        output = model(batch) # forward pass
    loss = loss_fn(
        output['estimates'],
        batch['source_magnitudes']
    )    
    loss_vals = {
        'L1Loss': loss.item(), 
        'loss': loss.item()
    }
    return loss_vals


def train(output_folder, epoch_length, max_epochs):
    global MAX_MIXTURES
    MAX_MIXTURES = int(1e8) # We'll set this to some impossibly high number for on the fly mixing.
    global stft_params
    stft_params = nussl.STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')
    train_folder = "~/.nussl/tutorial/train"
    val_folder = "~/.nussl/tutorial/valid"

    tfm = nussl_tfm.Compose([
        nussl_tfm.SumSources([['bass', 'drums', 'other']]),
        nussl_tfm.MagnitudeSpectrumApproximation(),
        nussl_tfm.IndexSources('source_magnitudes', 1),
        nussl_tfm.ToSeparationModel(),
    ])

    train_data = data.on_the_fly(stft_params, transform=tfm, 
        fg_path=train_folder, num_mixtures=MAX_MIXTURES, coherent_prob=1.0)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, num_workers=1, batch_size=10)

    val_data = data.on_the_fly(stft_params, transform=tfm, 
        fg_path=val_folder, num_mixtures=10, coherent_prob=1.0)
    val_dataloader = torch.utils.data.DataLoader(
        val_data, num_workers=1, batch_size=10)
    
    nf = stft_params.window_length // 2 + 1
    global model
    model = MaskInference.build(nf, 1, 50, 1, True, 0.0, 1, 'sigmoid')
    global optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    global loss_fn
    loss_fn = nussl.ml.train.loss.L1Loss()

    # Create the engines
    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=DEVICE
    )

    # Adding handlers from nussl that print out details about model training
    # run the validation step, and save the models.
    nussl.ml.train.add_stdout_handler(trainer, validator)
    nussl.ml.train.add_validate_and_checkpoint(output_folder, model, 
        optimizer, train_data, trainer, val_dataloader, validator)

    trainer.run(
        train_dataloader, 
        epoch_length, 
        max_epochs
    )


def evaluate(output_folder, separator):
    tfm = nussl_tfm.Compose([
        nussl_tfm.SumSources([['bass', 'drums', 'other']]),
    ])

    test_dataset = nussl.datasets.MUSDB18(subsets=['test'], transform=tfm)

    # Just do 5 items for speed. Change to 50 for actual experiment.
    for i in range(5):
        item = test_dataset[i]
        separator.audio_signal = item['mix']
        estimates = separator()

        source_keys = list(item['sources'].keys())
        estimates = {
            'vocals': estimates[0],
            'bass+drums+other': item['mix'] - estimates[0]
        }

        sources = [item['sources'][k] for k in source_keys]
        estimates = [estimates[k] for k in source_keys]

        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, source_labels=source_keys
        )
        scores = evaluator.evaluate()
        output_folder = Path(output_folder).absolute()
        output_folder.mkdir(exist_ok=True)
        output_file = output_folder / sources[0].file_name.replace('wav', 'json')
        with open(output_file, 'w') as f:
            json.dump(scores, f, indent=4)
    
    json_files = glob.glob(f"*.json")
    df = nussl.evaluation.aggregate_score_files(json_files, aggregator=np.nanmedian)
    nussl.evaluation.associate_metrics(separator.model, df, test_dataset)
    report_card = nussl.evaluation.report_card(df, report_each_source=True)
    print(report_card)
    

def convert_output_to_wav(separator):
    test_folder = "~/.nussl/tutorial/test/"
    test_data = data.mixer(stft_params, transform=None, 
        fg_path=test_folder, num_mixtures=MAX_MIXTURES, coherent_prob=1.0)
    item = test_data[0]

    separator.audio_signal = item['mix']
    estimates = separator()
    # Since our model only returns one source, let's tack on the
    # residual (which should be accompaniment)
    estimates.append(item['mix'] - estimates[0])
    stem1 = estimates[0]
    stem2 = estimates[1]
    stem1.write_audio_to_file('stem1.wav')
    stem2.write_audio_to_file('stem2.wav')


if __name__ == "__main__":
    """
    TO-DOs:
    1. Add in a consistent random seed for data to get the same output
    2. Add code to plot / store loss values
    3. Change for loop in evaluate() to loop through all 50 test songs
    """
    utils.logger()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset_path = str(Path.home()) + '/.nussl/tutorial'
    # Download dataset if it hasn't been downloaded
    if os.path.isdir(dataset_path) == False:
        data.prepare_musdb(dataset_path)

    output_folder = Path('.').absolute()
    train(output_folder, epoch_length=10, max_epochs=1)

    separator = nussl.separation.deep.DeepMaskEstimation(
        nussl.AudioSignal(), model_path='checkpoints/best.model.pth',
        device=DEVICE,
    )
    evaluate(output_folder, separator)
    convert_output_to_wav(separator)
