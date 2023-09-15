from CIA.positional_embeddings.positional_embedding import BasePositionalEmbedding
from torch import nn
from CIA.utils import flatten
import torch
import math


class SinusoidalElapsedTimeEmbedding(BasePositionalEmbedding):
    def __init__(self, positional_embedding_size, num_channels,
                 dataloader_generator, dropout, expand_channels, **kwargs):
        super(SinusoidalElapsedTimeEmbedding, self).__init__(expand_channels=expand_channels)
        assert positional_embedding_size % 2 == 0
        self.dataloader_generator = dataloader_generator
        self.positional_embedding_size = positional_embedding_size

        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_channels = num_channels
        self.mask_positions = kwargs['mask_positions']
        if self.mask_positions:
            self.mask_vector = nn.Parameter(
                torch.randn((self.positional_embedding_size, ))
            )

    def forward(self, x_embed, i=0, h=None, metadata_dict={}):
        assert i == 0
        if h is None:
            h = torch.zeros((x_embed.size(0),)).to(x_embed.device)
        assert 'original_sequence' in metadata_dict, (
            'Dictionnary metadata_dict must contain entry "original_sequence" in order to compute the elapsed time'
        )
        # Original sequence is in prefix order!
        x = metadata_dict['original_sequence']
        batch_size, num_events, num_channels = x.size()

        elapsed_time = self.dataloader_generator.get_elapsed_time(x)

        h = elapsed_time[:, -1]
        # if prefix mode
        if 'decoding_start' in metadata_dict:
            if elapsed_time.size(1) >= metadata_dict['decoding_start']:
                h = h - elapsed_time[:, metadata_dict['decoding_start'] - 1]

        # add zeros
        elapsed_time = torch.cat(
            [
                torch.zeros_like(elapsed_time)[:, :1],
                elapsed_time[:, :-1]
            ],
            dim=1
        )

        # check if in prefix mode:
        if 'decoding_start' in metadata_dict:
            if elapsed_time.size(1) > metadata_dict['decoding_start']:
                # we need to have an offset for the generated inpainted region
                elapsed_time[:, metadata_dict['decoding_start']:] = (
                    elapsed_time[:, metadata_dict['decoding_start']:] -
                    elapsed_time[:, metadata_dict['decoding_start']
                                 ].unsqueeze(1)
                )

        # add embedding_dim to elapsed time
        elapsed_time = elapsed_time.unsqueeze(2)

        # TODO scale?! only 10?!
        elapsed_time = elapsed_time * 100
        h = h * 100

        pe = torch.zeros(batch_size,
                         num_events, self.positional_embedding_size)
        pe = pe.to(device=x.device)

        div_term = torch.exp(torch.arange(0, self.positional_embedding_size, 2).float() * (
            -math.log(10000.0) / self.positional_embedding_size))
        div_term = div_term.to(device=x.device)
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        pe[:, :, 0::2] = torch.sin(elapsed_time * div_term)
        pe[:, :, 1::2] = torch.cos(elapsed_time * div_term)
        
        
        if self.expand_channels:
            pos_embedding = pe.repeat_interleave(
                self.num_channels, dim=1
            )
        else:
            pos_embedding = pe

        if self.mask_positions:
            if not self.expand_channels:
                raise NotImplementedError
            masked_positions = metadata_dict['masked_positions']
            flattened_masked_positions = flatten(masked_positions)
            flattened_masked_positions = flattened_masked_positions.view(
                batch_size * num_events * num_channels)
            pos_embedding = pos_embedding.view(
                batch_size * num_events * num_channels, self.positional_embedding_size
            )
            pos_embedding[flattened_masked_positions.bool(
            )] = self.mask_vector.unsqueeze(0)
            pos_embedding = pos_embedding.view(batch_size,
                                               num_events * num_channels,
                                               self.positional_embedding_size)

        pos_embedding = self.dropout(pos_embedding)
        x_embed = torch.cat([x_embed, pos_embedding], dim=2)
        return x_embed, h

    def forward_step(self, x, i=0, h=None, metadata_dict={}):
        if not self.expand_channels:
            raise NotImplementedError

        # time_shift must be the last feature
        assert self.dataloader_generator.features.index(
            'time_shift') == len(self.dataloader_generator.features) - 1

        assert 'original_token' in metadata_dict, (
            'Dictionnary metadata_dict must contain entry "original_token" in order to compute the elapsed time'
        )

        batch_size = x.size(0)
        # h represents the elapsed time
        if h is None:
            h = torch.zeros((batch_size, )).to(x.device)

        elapsed_time = h.unsqueeze(1)

        pe = torch.zeros(batch_size,
                         self.positional_embedding_size)
        pe = pe.to(device=x.device)

        div_term = torch.exp(torch.arange(0, self.positional_embedding_size, 2).float() * (
            -math.log(10000.0) / self.positional_embedding_size))
        div_term = div_term.to(device=x.device)
        div_term = div_term.unsqueeze(0)
        pe[:, 0::2] = torch.sin(elapsed_time * div_term)
        pe[:, 1::2] = torch.cos(elapsed_time * div_term)

        # dropout only on pe
        pe = self.dropout(pe)
        x_embed = torch.cat([x, pe], dim=1)

        # update h if the current token is a time_shift:
        if i % self.num_channels == self.num_channels - 1:
            # add fake features so that we can call get_elapsed_time
            target = metadata_dict['original_token']
            target = target.unsqueeze(1).unsqueeze(1)
            target = target.repeat(1, 1, self.num_channels)
            elapsed_time = self.dataloader_generator.get_elapsed_time(
                target
            )
            elapsed_time = elapsed_time.squeeze(1)

            # TODO scale?!
            elapsed_time = elapsed_time * 100

            h = h + elapsed_time

        # TODO check this
        if 'decoding_start' in metadata_dict:
            if i == self.num_channels * metadata_dict['decoding_start'] - 1:
                h = torch.zeros_like(h)

        return x_embed, h
