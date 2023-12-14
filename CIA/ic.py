from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self
import einops

import torch
import numpy as np
from CIA.dataloaders.dataloader import DataloaderGenerator

from CIA.dataset_managers.piano_midi_dataset import PianoMidiDataset

@dataclass
class ICCurve(Callable[[torch.FloatTensor], torch.FloatTensor]):
    def __call__(self, t : torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError
@dataclass
class DrawnICCurve(ICCurve):
    relative_time: bool
    def set_placeholder_length(self, placholder_length : float):
        if self.relative_time:
            self._placeholder_scale = 1 / placholder_length
    def __post_init__(self):
        self._placeholder_scale = 1.
@dataclass
class LinearInterpolation(DrawnICCurve):
    timepoints: List[float]
    ics: List[List[float]]
    def __post_init__(self):
        super().__post_init__()
        assert len(self.timepoints) == len(self.ics)
        self._timepoints = np.array(self.timepoints)
        self._ics = np.array(self.ics)
    def __call__(self, t : torch.FloatTensor) -> torch.FloatTensor:
        t = (t* self._placeholder_scale).numpy()
        return torch.tensor(np.stack([np.interp(t, self._timepoints, ics) for ics in self._ics.T], axis=1))[None]
@dataclass
class Piecewise(DrawnICCurve):
    # NOTE alternatively scipy.interpolate.interp1d(x, y, kind='nearest'), but it's deprecated
    timepoints: List[float]
    ics: List[List[float]]
    # time_relative: bool = False
    def __post_init__(self):
        super().__post_init__()
        assert len(self.timepoints) == len(self.ics) - 1
        self._timepoints = np.array(self.timepoints)
        self._ics = np.array(self.ics)
    def __call__(self, t : torch.FloatTensor) -> torch.FloatTensor:
        # Define the conditions for each step
        t = (t* self._placeholder_scale).numpy()
        timepoints = self._timepoints
        conditions = [t < timepoints[0]]
        for i in range(len(timepoints) - 1):
            conditions.append((t >= timepoints[i]) & (t < timepoints[i+1]))
        conditions.append(t >= timepoints[-1])
        return torch.tensor(np.stack([np.piecewise(t, conditions, self._ics[:,i]) for i in range(self._ics.shape[1])],axis=-1)[None])


@dataclass
class TimepointsGenerator(Callable[[torch.FloatTensor, dict], torch.FloatTensor]):
    # interface for extracting the timepoints for which the IC needs to be computed
    def __call__(self, piece: torch.LongTensor, metadata_dict: dict) -> torch.FloatTensor:
        raise NotImplementedError

@dataclass
class FixedStepTimepoints(TimepointsGenerator):
    step : float
    def __call__(self, piece: torch.LongTensor, metadata_dict: dict) -> torch.FloatTensor:
        placeholder_duration = metadata_dict['placeholder_duration'].item()
        return torch.arange(0, placeholder_duration, self.step)
    



@dataclass
class Interpolator(ICCurve):
    metric_times : Iterable[torch.FloatTensor]
    metric: Iterable[torch.FloatTensor] 
    weight_fn: Callable[[torch.FloatTensor], torch.FloatTensor]
    def __post_init__(self):
        lens = [len(ic) for ic in self.metric]
        assert all(l == len(ic) for l, ic in zip(lens, self.metric))
        self.metric_times = torch.nn.utils.rnn.pad_sequence(self.metric_times, batch_first=True)[..., None]
        self.metric = torch.nn.utils.rnn.pad_sequence(self.metric, batch_first=True)
        assert self.metric_times.dim() == 4 and self.metric.dim() == 3 # bz, tokens, channels, (t=1?)

    def __call__(self, t : torch.FloatTensor) -> torch.FloatTensor:
        #time_diffs = t[None, :, None] - self.ic_times[:, None]
        time_diffs = einops.rearrange(t, '(bz tok chan t) -> bz tok chan t', bz=1, chan=1, tok=1) - self.metric_times
        w = self.weight_fn(time_diffs)
        w[time_diffs <.0] = 0.
        # NOTE: the ic padding cancels automatically.
        return einops.einsum(w, self.metric, 'bz tok chan t, bz tok chan -> bz t chan')

@dataclass
class Weight(Callable[[torch.FloatTensor], torch.FloatTensor]):
    def __call__(self, time_diffs : torch.FloatTensor) -> torch.Tensor:
        raise NotImplementedError

@dataclass
class MovingAverage(Weight):
    window_size : float
    c: Union[float, List[Union[float, str]]]
    channel_weight: Union[float, List[Union[float, str]]] = 1.
    def __post_init__(self):
        if isinstance(self.c, List):
            self.c = [float(c_) for c_ in self.c]
        elif isinstance(self.c, float):
            self.c = [self.c]
        if isinstance(self.channel_weight, List):
            self.channel_weight = [float(a) for a in self.channel_weight]
        elif isinstance(self.channel_weight, float):
            self.channel_weight = [self.channel_weight]
        self.c_ = torch.tensor(self.c)[None, None, :, None] # bz=1, tok=1, channels
        self.cw = torch.tensor(self.channel_weight)[None, None, :, None] # bz=1, tok=1, channels
    def __call__(self, time_diffs : torch.FloatTensor) -> torch.Tensor:
        # NOTE: (bz, T, channels, tokens)
        # NOTE: numerical stability for 0 * inf
        e = 1e-9
        mov_avg = self.cw*(-self.c_*(time_diffs+e)).exp()
        mov_avg[time_diffs>=self.window_size] = 0
        return mov_avg


@dataclass
class Piece:
    path: str
    start_node: int
    n_inpaint: int

@dataclass
class ICRes:
    tok: torch.Tensor
    ic_tok: torch.Tensor
    entr_tok: torch.Tensor
    timepoints: torch.Tensor
    ic_int: torch.Tensor
    timepoints_int: torch.Tensor
    decoding_end: int
    piece: Piece
    ic_dev: Optional[torch.Tensor] = None
    def write(self, p : Path):
        torch.save(obj=self, f=p)
    @classmethod
    def load(cls, p : Path) -> Self:
        return torch.load(p)
def unique_timepoints(onsets : torch.Tensor, ics : torch.Tensor):
    assert ics.dim() == 2 and onsets.dim() == 1, "Batched not implimented..."
    timepoints = []
    for onset, ic in zip(onsets, ics):
        onset = onset.item()
        if timepoints and timepoints[-1][0] == onset:
            timepoints[-1][1] = timepoints[-1][1] + ic
        else:
            timepoints.append([onset, ic])
    unique_timepoint, cum_ics = zip(*timepoints)
    unique_timepoint = torch.tensor(unique_timepoint)
    cum_ics = torch.stack(cum_ics, dim=0)
    return unique_timepoint, cum_ics
def numerial_stable_softmax_entr(logits, dim=-1):
    p = torch.nn.functional.softmax(logits, dim=dim)
    return -(p * torch.nn.functional.log_softmax(logits, dim=dim))

@dataclass
class Data(Iterable[Tuple[torch.LongTensor, str, int, Optional[Piece]]]):
    label: str
    @property
    def dataloader_generator(self) -> DataloaderGenerator:
        return self.dataloader_generator_
    @dataloader_generator.setter
    def dataloader_generator(self, val) -> None:
        self.dataloader_generator_ = val
    def __len__(self) -> int:
        raise NotImplementedError
    def __iter__(self):
        raise NotImplementedError


@dataclass
class DataCache(Data):
    # dataloader_generator : DataloaderGenerator
    n_inpaint : Union[int, float]
    split : str
    midi_path : str = field(repr=False)
    cache_path : str = field(repr=False)
    n_pieces: Optional[int] = None
    def __post_init__(self):
        assert Path(self.midi_path).is_dir()
        if isinstance(self.n_inpaint, int):
            assert self.n_inpaint < 512 - 5
        elif not isinstance(self.n_inpaint, float):
            raise NotImplementedError
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
            yield original_x['x'], piece_name, self.n_inpaint, piece
    def __len__(self) -> int:
        return len(self.dataloader_generator.dataset) if self.n_pieces is None else self.n_pieces
        
@dataclass
class DataPiece(Data):
    pieces : Iterable[Piece]
    def __len__(self) -> int:
        return len(self.pieces)
    def __repr__(self) -> str:
        return f'DataPiece({self.label})'
    def __iter__(self):
        for piece in self.pieces:
            piece_name = Path(piece.path).stem + f'_start_{piece.start_node}_nodes_{piece.n_inpaint}'
            # NOTE: parallelize over number of samples per piece
            ds : PianoMidiDataset = self.dataloader_generator.dataset
            sequence = ds.process_score(piece.path)
            orig_seq_length = len(sequence['pitch'])
            if piece.start_node > 0:
                sequence = {k : v[piece.start_node:] for k,v in sequence.items()}
            # if piece.start_node < 0:
            #     raise NotImplementedError('Tetst that indeed it works')
            sequence = ds.add_start_end_symbols(
                sequence, start_time=piece.start_node, sequence_size=ds.sequence_size
            )
            # Tokenize
            sample = ds.tokenize(sequence)
            x = torch.tensor([sample[e] for e in self.dataloader_generator.features])
            original_x = einops.rearrange(x, 'f n -> 1 n f')
            yield original_x, piece_name, piece.n_inpaint, piece
        # raise StopIteration
@dataclass
class Experiment:
    time_points_generator: TimepointsGenerator
    weight : Weight
    dataset : Data
    ic_curve: Optional[ICCurve]
    match_metric: str = 'ic'
    # NOTE: here we should have either the test set, or some named collection of pieces....
    def __post_init__(self):
        assert self.match_metric in ['ic', 'typicality']


@dataclass
class SamplingConfig:
    k_traces: int
    temperature: float