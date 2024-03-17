from dataclasses import InitVar, dataclass, field
from datetime import datetime
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
from warnings import warn
import einops

import torch
import numpy as np
# NOTE: as these two classes are used only for typing reasons, we could probably get rid of them...
# However, since the  functionality is tighly coupled to using these constructs. It seems, we still need it.
from CIA.dataloaders.dataloader import DataloaderGenerator
from CIA.dataset_managers.piano_midi_dataset import PianoMidiDataset
from pretty_midi import PrettyMIDI



        # return torch.tensor([[self.best_times[-1]]])



@dataclass
class Piece:
    path: str
    start_node: Union[int, float, str]
    n_inpaint: Union[int, float, str]
    # end_window: InitVar[Optional[Union[int, float, str]]] = None
    end_window: Optional[Union[int, float, str]] = None
    _end_window: Optional[Union[int, float, str]] = field(init=False, repr=False)
    # def __post_init__(self, end_window):
    def __post_init__(self):
        bot = datetime(1900,1,1)
        if isinstance(self.start_node, str):
            self.start_node = (datetime.strptime(self.start_node, '%M:%S.%f') - bot).total_seconds()
        if isinstance(self.n_inpaint, str):
            self.n_inpaint = (datetime.strptime(self.n_inpaint, '%M:%S.%f') - bot).total_seconds()
        # self._end_window = end_window
    # NOTE: lazy evaluation of end_window
    @property
    def end_window(self) -> int:
        bot = datetime(1900,1,1)
        if isinstance(self._end_window, str):
            self._end_window = (datetime.strptime(self._end_window, '%M:%S.%f') - bot).total_seconds()
        if isinstance(self.start_node, float) and isinstance(self.n_inpaint, float):
            midi = PrettyMIDI(self.path)
            notes_mozart = midi.instruments[0].notes
            sorted(notes_mozart, key=lambda n: n.start)
            start_time = self.start_node
            inpaint_end_time = start_time + self.n_inpaint
            start_decoding = np.argmin([abs(n.end - start_time) for n in  notes_mozart])
            end_decoding = np.argmin([abs(n.end - inpaint_end_time) for n in  notes_mozart])
            warn('hard coded n_begin_notes')
            n_begin_notes = 256
            self.start_node = start_decoding - n_begin_notes
            self.n_inpaint = end_decoding - start_decoding
            if self._end_window is not None:
                end_window_time = inpaint_end_time + self._end_window
                end_window = np.argmin([abs(n.end - end_window_time) for n in  notes_mozart])
                self._end_window = end_window - end_decoding
        return self._end_window
    @end_window.setter
    def end_window(self, end_window : Optional[int]):
        self._end_window = end_window

    @property
    def name(self) -> str:
        # TODO: hacky solution to making sure that durations have been converted to number of notes.
        end_window = self.end_window
        return Path(self.path).stem + f'_start_{self.start_node}_nodes_{self.n_inpaint}' + ('' if end_window is None else f'_end_{self.end_window}')
# def unique_timepoints(onsets : torch.Tensor, ics : torch.Tensor):
#     assert ics.dim() == 2 and onsets.dim() == 1, "Batched not implimented..."
#     timepoints = []
#     for onset, ic in zip(onsets, ics):
#         onset = onset.item()
#         if timepoints and timepoints[-1][0] == onset:
#             timepoints[-1][1] = timepoints[-1][1] + ic
#         else:
#             timepoints.append([onset, ic])
#     unique_timepoint, cum_ics = zip(*timepoints)
#     unique_timepoint = torch.tensor(unique_timepoint)
#     cum_ics = torch.stack(cum_ics, dim=0)
#     return unique_timepoint, cum_ics
@dataclass
class Data(Iterable[Tuple[torch.LongTensor, str, int, Optional[Piece]]]):
    label: str
    @property
    def dataloader_generator(self) -> DataloaderGenerator:
    # def dataloader_generator(self):
        return self.dataloader_generator_
    @dataloader_generator.setter
    def dataloader_generator(self, val) -> None:
        self.dataloader_generator_ = val
    def __len__(self) -> int:
        raise NotImplementedError
    def __iter__(self) -> Tuple[torch.Tensor, str, int, Piece]:
        raise NotImplementedError


@dataclass
class DataCache(Data):
    # dataloader_generator : DataloaderGenerator
    n_inpaint : Union[int, float]
    split : str
    midi_path : str = field(repr=False)
    cache_path : str = field(repr=False)
    n_pieces: Optional[int] = None
    end_window: Optional[Union[int, float]] = None
    def __post_init__(self):
        # assert Path(self.midi_path).is_dir()
        if isinstance(self.n_inpaint, int):
            assert self.n_inpaint < 512 - 5
        elif not isinstance(self.n_inpaint, float):
            raise NotImplementedError
        # if self.end_window is not None:
        #     raise NotImplementedError
        os.environ['PIA_MIDI_PATH'] = self.midi_path
        os.environ['PIA_CACHE_PATH'] = self.cache_path
    @Data.dataloader_generator.setter
    def dataloader_generator(self, val) -> None:
        # super().dataloader_generator.fset(self, val)
        # super(DataCache, self).dataloader_generator.fset(self, val)
        self.dataloader_generator_ = val
        ret = self.dataloader_generator.dataloaders(batch_size = 1, shuffle_val=True)
        val.dataset.split = self.split 
        dl_idx = {
            'train': 0,
            'validation': 1,
            'test': 2
        }[self.split]
        self.dataloader = ret[dl_idx]
    def __iter__(self):
        for i, original_x in enumerate(self.dataloader):
            if i == self.n_pieces:
                return
            piece_name = str(i)
            piece = None
            yield original_x['x'], piece_name, self.n_inpaint, self.end_window, piece
    def __len__(self) -> int:
        return len(self.dataloader_generator.dataset) if self.n_pieces is None else self.n_pieces
        
@dataclass
class DataPiece(Data):
    pieces : Iterable[Piece]
    # TODO: deprecate
    cache_path : str = field(repr=False)
    def __post_init__(self):
        os.environ['PIA_CACHE_PATH'] = self.cache_path
    def __len__(self) -> int:
        return len(self.pieces)
    def __repr__(self) -> str:
        return f'DataPiece({self.label})'
    def __iter__(self):
        ds : PianoMidiDataset = self.dataloader_generator.dataset
        # ds = self.dataloader_generator.dataset
        for piece in self.pieces:
            piece_name = piece.name
            # NOTE: parallelize over number of samples per piece
            sequence = ds.process_score(piece.path)
            orig_seq_length = len(sequence['pitch'])
            start_node = piece.start_node if piece.start_node > 0 else 0
            # NOTE: this is available in data_processor.num_events_before
            warn('hard coded num_before tokens')
            n_begin_notes = 256
            if piece.start_node > 0:
                # raise NotImplementedError('Test that the following line works by also when piece.start <=0')
                warn('Test that the following line works by also when piece.start <=0')
            end_node = piece.start_node + n_begin_notes + piece.n_inpaint + piece.end_window if piece.end_window is not None else None
            sequence = {k : v[slice(start_node, end_node)] for k,v in sequence.items()}
            # if piece.start_node < 0:
            #     raise NotImplementedError('Tetst that indeed it works')
            warn('This put\'s the first token of ')
            sequence = ds.add_start_end_symbols(
                sequence, start_time=piece.start_node, sequence_size=ds.sequence_size
            )
            # Tokenize
            sample = ds.tokenize(sequence)
            x = torch.tensor([sample[e] for e in self.dataloader_generator.features])
            original_x = einops.rearrange(x, 'f n -> 1 n f')
            yield original_x, piece_name, piece.n_inpaint, piece.end_window, piece
        # raise StopIteration
