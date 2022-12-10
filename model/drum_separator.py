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


def train(output_folder, batch_size, max_epochs, epoch_length):    
    train_folder = "~/.nussl/tutorial/train"
    val_folder = "~/.nussl/tutorial/valid"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tfm = nussl_tfm.Compose([
        #nussl_tfm.SumSources([['bass', 'drums', 'other']]), # unused - vocal separation
        nussl_tfm.SumSources([['bass', 'other', 'vocals']]), # drum separation
        nussl_tfm.MagnitudeSpectrumApproximation(),
        nussl_tfm.IndexSources('source_magnitudes', 1), # 'drums' comes after 'bass+other+vocals' alphabetically
        nussl_tfm.ToSeparationModel(),
    ])

    # TODO: check if there's a way to mix data consistently with a random seed
    train_data = data.on_the_fly(stft_params, transform=tfm, 
        fg_path=train_folder, num_mixtures=MAX_MIXTURES, coherent_prob=1.0)
    # TODO: understand what this function does and what would be a suitable batch size to use
    train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=1, batch_size=batch_size)

    val_data = data.on_the_fly(stft_params, transform=tfm, 
        fg_path=val_folder, num_mixtures=10, coherent_prob=1.0)
    val_dataloader = torch.utils.data.DataLoader(val_data, num_workers=1, batch_size=batch_size)
    
    nf = stft_params.window_length // 2 + 1
    global model
    model = MaskInference.build(nf, 1, 50, 1, True, 0.0, 1, 'sigmoid')
    model = model.to(DEVICE)
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
        max_epochs=max_epochs,
        epoch_length=epoch_length
    )


def evaluate(separator, output_path):
    tfm = nussl_tfm.Compose([
        #nussl_tfm.SumSources([['bass', 'drums', 'other']]), unused - for vocal separation
        #nussl_tfm.SumSources([['drums', 'other', 'vocals']]), unused - for bass separation
        nussl_tfm.SumSources([['bass', 'other', 'vocals']]), # for drum separation
    ])

    test_dataset = nussl.datasets.MUSDB18(subsets=['test'], transform=tfm)
    output_folder = Path(output_path).absolute()
    output_folder.mkdir(exist_ok=True)

    # TODO: change the for loop to loop through the whole test dataset when doing final evaluations
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        separator.audio_signal = item['mix']
        filename = item['mix'].file_name
        estimates = separator()

        # source_keys = ['vocals', 'bass+drums+other'] (have the same keys as the estimates dictionary)
        source_keys = list(item['sources'].keys())
        estimates = {
            'drums': estimates[0],
            'bass+other+vocals': item['mix'] - estimates[0]
        }

        # write audio output to wav
        estimates['drums'].write_audio_to_file(output_path + filename + '_drum.wav')
        estimates['bass+other+vocals'].write_audio_to_file(output_path + filename + '_other.wav')

        sources = [item['sources'][k] for k in source_keys]
        estimates = [estimates[k] for k in source_keys]

        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, source_labels=source_keys
        )
        scores = evaluator.evaluate()
        output_file = output_folder / sources[0].file_name.replace('wav', 'json')
        with open(output_file, 'w') as f:
            json.dump(scores, f, indent=4)
    
    json_files = glob.glob(str(output_folder) + "/*.json")
    df = nussl.evaluation.aggregate_score_files(json_files, aggregator=np.nanmedian)
    nussl.evaluation.associate_metrics(separator.model, df, test_dataset)
    report_card = nussl.evaluation.report_card(df, report_each_source=True)
    print(report_card)
    

def plot_validation_loss(filepath, output_path):
    model_checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    # print("trainer.state_dict")
    # print(model_checkpoint['metadata']['trainer.state_dict'])
    loss_history = model_checkpoint['metadata']['trainer.state.epoch_history']['validation/L1Loss']
    plt.plot(loss_history)
    plt.xlabel('# of Epochs')
    plt.ylabel('Validation loss')
    plt.title('Validation Loss of Drum Separator Model')
    plt.savefig(output_path + 'validation_loss.png')


if __name__ == "__main__":
    utils.logger()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_MIXTURES = int(1e8) # We'll set this to some impossibly high number for on the fly mixing.
    stft_params = nussl.STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')
    output_path = 'drum_output/'
    output_folder = Path(output_path).absolute()
    
    dataset_path = str(Path.home()) + '/.nussl/tutorial'
    # Download dataset if it hasn't been downloaded
    if os.path.isdir(dataset_path) == False:
        data.prepare_musdb(dataset_path)

    # batch_size = number of training examples in a batch
    # max_epoch = total number of epochs ran in training
    # epoch_length = number of batches in one epoch
    # train(output_folder, batch_size=10, max_epochs=30, epoch_length=20)

    checkpoint_path = output_path + 'checkpoints/best.model.pth'
    separator = nussl.separation.deep.DeepMaskEstimation(
        nussl.AudioSignal(), model_path=checkpoint_path,
        device=DEVICE,
    )

    plot_validation_loss(checkpoint_path, output_path)
    evaluate(separator, output_path)
