from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Type
from typing_extensions import Self
import einops

import torch


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
class Interpolator(Callable[[torch.FloatTensor], torch.FloatTensor]):
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
class ICRes:
    tok: torch.Tensor
    ic_tok: torch.Tensor
    ic_int: torch.Tensor
    timepoints: torch.Tensor
    decoding_end: Optional[int] = None

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