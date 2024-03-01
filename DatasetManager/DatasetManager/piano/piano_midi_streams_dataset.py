import glob
import itertools
import math
import os
import pickle
import random
import shutil
import copy
import numpy as np
import pretty_midi
import torch
import re
from torch.utils import data
from tqdm import tqdm

from DatasetManager.piano.piano_helper import extract_cc, find_nearest_value
from DatasetManager.piano.piano_midi_dataset import PianoMidiDataset

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning: 
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end: 
p0 p1 TS p0 p1 p2 TS p0 END STOP X X X

"""

START_SYMBOL = 'START'
END_SYMBOL = 'END'
PAD_SYMBOL = 'XX'


class PianoMidiStreamsDataset(PianoMidiDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 sequence_size,
                 smallest_time_shift,
                 max_transposition,
                 time_dilation_factor,
                 velocity_shift,
                 transformations
                 ):
        """
        All transformations
        {
            'time_shift': True,
            'time_dilation': True,
            'transposition': True
        }

        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        super().__init__(corpus_it_gen,
                         sequence_size,
                         smallest_time_shift,
                         max_transposition,
                         time_dilation_factor,
                         velocity_shift,
                         transformations)
        return

    def __str__(self):
        prefix = str(self.corpus_it_gen)
        name = f'PianoMidiStreams-' \
               f'{prefix}-' \
               f'{self.sequence_size}_' \
               f'{self.smallest_time_shift}'
        return name

    @property
    def dataset_file(self):
        dataset_dir = f'{self.cache_dir}/{str(self)}'
        return dataset_dir

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        THIS CHANGES; see chris
        return x, index, time_dilation, velocity_dilation