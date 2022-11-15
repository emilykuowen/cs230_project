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
            connections.append([key, f'model:{key}'])
        
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

def train_step(batch):
    optimizer.zero_grad()
    output = model(batch) # forward pass
    loss = loss_fn(
        output['estimates'],
        batch['source_magnitudes']
    )
    loss.backward() # backwards + gradient step
    optimizer.step()

    return loss.item() # return the loss for bookkeeping.
""" def train_step(engine, batch):
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
    
    return loss_vals """
    
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



if __name__ == "__main__":
    # Prepare MUSDB
    data.prepare_musdb('~/.nussl/tutorial/')

    # Training Loop
    utils.logger()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_MIXTURES = int(1e8) # We'll set this to some impossibly high number for on the fly mixing.
    stft_params = nussl.STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')

    tfm = nussl_tfm.Compose([
    nussl_tfm.SumSources([['bass', 'drums', 'other']]),
    nussl_tfm.MagnitudeSpectrumApproximation(),
    nussl_tfm.IndexSources('source_magnitudes', 1),
    nussl_tfm.ToSeparationModel(),
    ])

    train_folder = "~/.nussl/tutorial/train"
    val_folder = "~/.nussl/tutorial/valid"

    train_data = data.on_the_fly(stft_params, transform=tfm, 
        fg_path=train_folder, num_mixtures=MAX_MIXTURES, coherent_prob=1.0)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, num_workers=1, batch_size=10)

    val_data = data.on_the_fly(stft_params, transform=tfm, 
        fg_path=val_folder, num_mixtures=10, coherent_prob=1.0)
    val_dataloader = torch.utils.data.DataLoader(
        val_data, num_workers=1, batch_size=10)

    nf = stft_params.window_length // 2 + 1
    model = MaskInference.build(nf, 1, 50, 1, True, 0.0, 1, 'sigmoid')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nussl.ml.train.loss.L1Loss()


    item = train_data[0] # Because of the transforms, this produces tensors.
    batch = {} # A batch of size 1, in this case. Usually we'd have more.
    for key in item:
        if torch.is_tensor(item[key]):
            batch[key] = item[key].float().unsqueeze(0)
    print(batch)
    print(len(batch))        

    N_ITERATIONS = 100
    loss_history = [] # For bookkeeping

    pbar = tqdm.tqdm(range(N_ITERATIONS))
    for _ in pbar:
        loss_val = train_step(batch)
        loss_history.append(loss_val)
        pbar.set_description(f'Loss: {loss_val:.6f}')

    plt.plot(loss_history)
    plt.xlabel('# of iterations')
    plt.ylabel('Training loss')
    plt.title('Train loss history of our model')
    plt.show()

    # Create the engines
    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=DEVICE
    )

    # We'll save the output relative to this notebook.
    output_folder = Path('.').absolute()

    # Adding handlers from nussl that print out details about model training
    # run the validation step, and save the models.
    nussl.ml.train.add_stdout_handler(trainer, validator)
    nussl.ml.train.add_validate_and_checkpoint(output_folder, model, 
        optimizer, train_data, trainer, val_dataloader, validator)

    trainer.run(
        train_dataloader, 
        epoch_length=10, 
        max_epochs=2
    )

    #print(model.config)
    #print(model)
    #model.config['modules']['model']['args']
    #print(model.config['modules']['model']['module_snapshot'])