import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import nussl
from nussl.ml.networks.modules import AmplitudeToDB
from nussl.ml.networks.modules import BatchNorm
from nussl.ml.networks.modules import RecurrentStack
from nussl.ml.networks.modules import Embedding

nussl.utils.seed(0)
to_numpy = lambda x: x.detach().numpy()
to_tensor = lambda x: torch.from_numpy(x).reshape(-1, 1).float()


class Model(nn.Module):
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
        return estimates


def print_stats(data):
    print(
        f"Shape: {data.shape}\n"
        f"Mean: {data.mean().item()}\n"
        f"Var: {data.var().item()}\n"
    )

if __name__ == "__main__":
    musdb = nussl.datasets.MUSDB18(download=True)
    item = musdb[40]

    representation = np.abs(item['mix'].stft())
    vocals_representation = np.abs(item['sources']['vocals'].stft())

    # fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    # ax[0].imshow(20 * np.log10(representation[..., 0]), origin='lower', aspect='auto')
    # ax[0].set_title('Mixture spectrogram')
    # ax[1].imshow(20 * np.log10(vocals_representation[..., 0]), origin='lower', aspect='auto')
    # ax[1].set_title('Vocals spectrogram')
    # plt.show()

    mask = vocals_representation / (np.maximum(vocals_representation, representation) + 1e-8)

    # fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    # ax[0].imshow(20 * np.log10(representation[..., 0]), origin='lower', aspect='auto')
    # ax[0].set_title('Mixture spectrogram')
    # ax[1].imshow(mask[..., 0], origin='lower', aspect='auto')
    # ax[1].set_title('Vocals mask')
    # plt.show()

    masked = mask * representation
    # fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    # ax[0].imshow(20 * np.log10(vocals_representation[..., 0]), origin='lower', aspect='auto')
    # ax[0].set_title('Actual vocals')
    # ax[1].imshow(20 * np.log10(masked[..., 0]), origin='lower', aspect='auto')
    # ax[1].set_title('Masked vocals')
    # plt.show()

    mix_phase = np.angle(item['mix'].stft())
    masked_stft = masked * np.exp(1j * mix_phase)
    new_signal = nussl.AudioSignal(stft=masked_stft, sample_rate=item['mix'].sample_rate)
    new_signal.istft()
    new_signal.embed_audio(display=False)

    mix_tensor = torch.from_numpy(representation)
    mask_tensor = torch.rand_like(mix_tensor.unsqueeze(-1))
    print(mix_tensor.shape, mask_tensor.shape)

    # masking operation:
    masked_tensor = mix_tensor.unsqueeze(-1) * mask_tensor
    print(masked_tensor.shape)
    
    print("Shape before (nf, nt, nac): ", mix_tensor.shape)
    mix_tensor = mix_tensor.permute(1, 0, 2).unsqueeze(0).float()
    print("Shape after (nb, nt, nf, nac):", mix_tensor.shape)
    
    nb, nt, nf, nac = mix_tensor.shape # not enough values to unpack, expected 4 but got 3
    model = Model(nf, nac, 50, 2, True, 0.3, 1, 'sigmoid')
    output = model(mix_tensor)

    print_stats(mix_tensor)
    print_stats(output)