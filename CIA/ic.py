from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Type
from typing_extensions import Self
import einops

import torch
import numpy as np

# def gen_interpolate(
#         ic_times : Iterable[torch.FloatTensor],
#         ics: Iterable[torch.FloatTensor],
#         weight_fn: Callable[[torch.FloatTensor], torch.FloatTensor]
#     ) ->  Callable[[torch.FloatTensor], torch.FloatTensor]:
#     lens = [len(ic) for ic in ics]
#     assert all(l == len(ic) for l, ic in zip(lens, ics))
#     ic_times = torch.nn.utils.rnn.pad_sequence(ic_times, batch_first=True)
#     ics = torch.nn.utils.rnn.pad_sequence(ics, batch_first=True)
#     def interpolate(t : torch.FloatTensor) -> torch.FloatTensor:
#         # NOTE: a matrix of (bz, T, TOKENS)
#         time_diffs = t[None, :, None] - ic_times[:, None]
#         w = weight_fn(time_diffs)
#         w[time_diffs <.0] = 0.
#         # NOTE: automatically the padding cancels automatically because of the ic padding.
#         return einops.einsum(w, ics, 'bz T tok, bz tok ... -> bz T ...')
#     return interpolate
@dataclass
class ICCurve(Callable[[torch.FloatTensor], torch.FloatTensor]):
    def __call__(self, t : torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError
@dataclass
class DrawnICCurve(ICCurve):
    def set_placeholder_length(self, placholder_length : float):
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
class Interpolator(ICCurve):
    ic_times : Iterable[torch.FloatTensor]
    ics: Iterable[torch.FloatTensor]
    weight_fn: Callable[[torch.FloatTensor], torch.FloatTensor]
    def __post_init__(self):
        lens = [len(ic) for ic in self.ics]
        assert all(l == len(ic) for l, ic in zip(lens, self.ics))
        self.ic_times = torch.nn.utils.rnn.pad_sequence(self.ic_times, batch_first=True)
        self.ics = torch.nn.utils.rnn.pad_sequence(self.ics, batch_first=True)

    def __call__(self, t : torch.FloatTensor) -> torch.FloatTensor:
        # NOTE: a matrix of (bz, T, TOKENS)
        time_diffs = t[None, :, None] - self.ic_times[:, None]
        w = self.weight_fn(time_diffs)
        w[time_diffs <.0] = 0.
        # NOTE: automatically the padding cancels automatically because of the ic padding.
        return einops.einsum(w, self.ics, 'bz T tok, bz tok ... -> bz T ...')

@dataclass
class MovingAverage(Callable[[torch.FloatTensor], torch.FloatTensor]):
    window_size : float
    c: float
    def __call__(self, time_diffs : torch.FloatTensor) -> torch.Tensor:
        # NOTE: (bz, T, tokens)
        mov_avg = (-self.c*time_diffs).exp()
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
    timepoints: torch.Tensor
    ic_int: torch.Tensor
    timepoints_int: torch.Tensor
    decoding_end: int
    piece: Piece
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

