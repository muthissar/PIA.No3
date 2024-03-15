from dataclasses import InitVar, dataclass, field
from datetime import datetime
import hashlib
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self
from warnings import warn
import einops

import torch
import numpy as np
from CIA.dataloaders.dataloader import DataloaderGenerator

from CIA.dataset_managers.piano_midi_dataset import PianoMidiDataset
from pretty_midi import PrettyMIDI
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
    def __call__(self, t : List[torch.FloatTensor]) -> List[torch.FloatTensor]:
        # t : bz, T 
        lens = [len(tt) for tt in t]
        t = torch.nn.utils.rnn.pad_sequence(t, batch_first=True, padding_value=-1)
        assert t.dim() == 2
        orig_shape  = t.shape
        t = (t* self._placeholder_scale).numpy()
        res = torch.FloatTensor(np.stack([np.interp(t.flatten(), self._timepoints, ics) for ics in self._ics.T], axis=-1))
        res = res.view(*orig_shape, -1)
        res = [r[:l] for r, l in zip(res, lens)]
        return res
        # res = torch.tensor(np.stack([np.interp(t* self._placeholder_scale, self._timepoints, ics) for ics in self._ics.T], axis=-1))
        # return res.view(*orig_shape, -1)

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

class TimepointsGenerator:
    def initialize(self, placholder_duration : float):
        raise NotImplementedError
    def update_is_exceeded(self, t : torch.FloatTensor, idx : int) -> bool:
        raise NotImplementedError
    def done(self) -> bool:
        raise NotImplementedError
    def update_step(self, idx :int) -> None:
        raise NotImplementedError
    def progress(self) -> Tuple[int, int]:
        raise NotImplementedError
    def get_eval_points(self):
        raise NotImplementedError
    def get_all_eval_points(self):
        raise NotImplementedError
# TODO: this is actually an iterator'(ish) thing
@dataclass
class FixedStepTimepoints(TimepointsGenerator):
    step : float
    eval_step: float
    k_traces : int = field(repr=False)
    tol_placeholder_duration : float = field(repr=False, default=0.2)
    def __post_init__(self):
        # super().__post_init__()
        assert (self.step / self.eval_step).is_integer()
    def initialize(self, placholder_duration : float):
        # self.current_step = 0   
        self.current_step = 1   
        self.placeholder_duration = placholder_duration
        self.next_time_traces = torch.zeros(self.k_traces)
    def update_is_exceeded(self, t : torch.FloatTensor, idx : int) -> bool:
        '''
        Times
        '''
        # return t >= self.current_step * self.step
        self.next_time_traces[idx] = t
        return t > self.current_step * self.step
    def done(self) -> bool:
        return self.current_step * self.step >= self.placeholder_duration
    def update_step(self, idx :int) -> None:
        # NOTE: IN case multiple steps are skipped
        assert self.next_time_traces[idx] > self.current_step * self.step
        new_step = int(self.next_time_traces[idx] // self.step)
        crossings = new_step - self.current_step
        if crossings > 1:
            warn(f'Skipped {crossings} steps, which ammounts to {crossings * self.step} seconds.')
        self.current_step = max(new_step, self.current_step+1)
    def progress(self) -> Tuple[int, int]:
        '''
        Returns the (rounded) current number of secs generated and the total number of secs to be generated
        '''
        return round(self.current_step * self.step), round(self.placeholder_duration)
    def get_eval_points(self):
        if self.current_step == 0:
            ts = self.k_traces*[torch.tensor([0.0])]
        else:
            ts = []
            for next_time_trace in self.next_time_traces:
                e = 1e-9
                # NOTE: overshoot next_time_trace - (self.current_step) * self.step
                ts_ = torch.arange((self.current_step-1) * self.step, next_time_trace-e, self.eval_step)
                # if len(ts_) == 0 or ts_[-1] != next_time_trace:
                #     ts_ = torch.cat([ts_, torch.tensor([next_time_trace])])
                ts.append(ts_)

            # ts = torch.arange((self.current_step-1) * self.step, self.current_step * self.step, self.eval_step)
        # return ts[None]
        return ts
    def get_all_eval_points(self):
        return torch.arange(0, self.placeholder_duration, self.eval_step)[None]
        
@dataclass
class SingleNoteTimepoints(TimepointsGenerator):
    # def __post_init__(self):
    #     # assert (self.step / self.eval_step).is_integer()
    #     self.best_times = [0.0]
    k_traces : int = field(repr=False)
    eval_step: float = field(repr=False, default=0.1)
    tol_placeholder_duration : float = field(repr=False, default=0.2)
    def initialize(self, placholder_duration : float):
        # self.current_time = 0.0
        self.placeholder_duration = placholder_duration
        # self.past_time_traces =  torch.zeros(self.k_traces)
        # self.current_time_traces =  torch.zeros(self.k_traces)
        self.next_time_traces = torch.zeros(self.k_traces)
        self.best_times = [0.0]
        warn('This value is hardcoded, and the same in decoder_events_handler.py')
    def update_is_exceeded(self, t : torch.FloatTensor, idx : int) -> bool:
        '''
        Times
        '''
        # self.current_time_traces[idx] = self.next_time_traces[idx]
        # TODO: the ics of shift should be set on the next note...
        self.next_time_traces[idx] = t
        return True
    def done(self) -> bool:
        return len(self.best_times)>0 and self.best_times[-1] >= self.placeholder_duration - self.tol_placeholder_duration
    def update_step(self, idx : int) -> None:
        # self.best_times.append(t.item())
        # raise NotImplementedError('This does not work if generation[idx] is done... ')
        self.best_times.append(self.next_time_traces[idx].item())
        # self.time_traces = []
    def progress(self) -> Tuple[int, int]:
        '''
        Returns the (rounded) current number of secs generated and the total number of secs to be generated
        '''
        return round(self.best_times[-1] if len(self.best_times) else 0.), round(self.placeholder_duration)
    def get_eval_points(self) -> List[torch.FloatTensor]:
        # if len(self.best_times) == 0:
        #     ts = torch.zeros(self.k_traces, 1)
        # else:
        #     ts = torch.tensor(self.time_traces)[:, None]
        # ts = self.current_time_traces[:, None]
        # ts = torch.tensor(self.best_times[-1])[None, None]
        # NOTE: quite inefficient, because in principle it would be enough to only use 
        # next_time_traces, because we always expand from the note before. However, for times where the sequence is done, 
        # we need to evaluate in the extra points. Can  we simply get rid of choosing the done sequences?
        # ts = torch.cat([torch.tensor(self.best_times)[None].expand(self.k_traces, -1), self.next_time_traces[:, None]], dim=1)
        # ts = self.next_time_traces[:, None]
        ts = []
        for next_time_trace in self.next_time_traces:
            ts_ = torch.arange(self.best_times[-1], next_time_trace, self.eval_step)
            if len(ts_) == 0 or ts_[-1] != next_time_trace:
                ts_ = torch.cat([ts_, torch.tensor([next_time_trace])])
            ts.append(ts_)
        return ts
    def get_all_eval_points(self):
        # return torch.arange(0, self.placeholder_duration, self.eval_step)
        return torch.tensor(self.best_times)[None]
        # return torch.tensor([[self.best_times[-1]]])



@dataclass
class Interpolator(ICCurve):
    metric_times : Iterable[torch.FloatTensor]
    metric: Iterable[torch.FloatTensor] 
    weight_fn: Callable[[torch.FloatTensor], torch.FloatTensor]
    metric_clip:  Optional[torch.FloatTensor]= None
    reduce_equal_times: str = 'sum' # 'max', 'mean'
    def __post_init__(self):
        lens = [len(ic) for ic in self.metric]
        assert all(l == len(ic) for l, ic in zip(lens, self.metric))
        self.metric_times = torch.nn.utils.rnn.pad_sequence(self.metric_times, batch_first=True)
        self.metric = torch.nn.utils.rnn.pad_sequence(self.metric, batch_first=True)
        if self.reduce_equal_times in ['max', 'mean']:
            times_t = einops.rearrange(self.metric_times, 'bz tok chan -> (bz chan) tok')
            ics_t = einops.rearrange(self.metric, 'bz tok chan -> (bz chan) tok')
            for times, ics in zip(times_t, ics_t):
                unique, inverse, count = times.unique( return_inverse=True, return_counts=True)
                for t,c in zip(unique, count):
                    time_mask = times==t
                    max_val, max_idx = ics[time_mask].max(dim=0)
                    # max
                    if self.reduce_equal_times == 'max':
                        ics[time_mask] = max_val/c
                    else:
                    # mean
                        ics[time_mask] /= c
        elif self.reduce_equal_times != 'sum':
            raise NotImplementedError

        self.metric_times = self.metric_times[..., None]
        # self.metric = min(self.metric, self.metric_cap)
        if self.metric_clip is None:
            warn('Most likely a cap needs to be set. For instance by calculating quantile .95')
        else:
            # torch.as_tensor(self.metric_clip, dtype=torch.float32)
            self.metric = torch.where(self.metric > self.metric_clip, self.metric_clip, self.metric)
        assert self.metric_times.dim() == 4 and self.metric.dim() == 3 # bz, tokens, channels, (t=1?)

    def __call__(self, t : List[torch.FloatTensor]) -> List[torch.FloatTensor]:
        #time_diffs = t[None, :, None] - self.ic_times[:, None]
        # time_diffs = einops.rearrange(t, '(bz tok chan t) -> bz tok chan t', bz=1, chan=1, tok=1) - self.metric_times
        lens = [len(tt) for tt in t]
        t = torch.nn.utils.rnn.pad_sequence(t, batch_first=True, padding_value=-1)
        assert t.dim() == 2 # bz, t
        time_diffs = t[:, None, None] - self.metric_times
        # rear = einops.rearrange(time_diffs, 'bz obs channels time_eval -> obs (bz channels time_eval)')
        # hehe = rear.unique(dim=0, return_counts=True)
        w = self.weight_fn(time_diffs, self.metric)
        w[time_diffs <.0] = 0.
        # NOTE: the ic padding cancels automatically.
        
        metric = self.metric.expand(w.shape[0], *self.metric.shape[1:])
        ret = einops.einsum(w, metric, 'bz tok chan t, bz tok chan -> bz t chan')
        # return ret
        return [r[:l] for r, l in zip(ret, lens)]

@dataclass
class Weight(Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]):
    def __call__(self, time_diffs : torch.FloatTensor, metric_vals : torch.FloatTensor) -> torch.Tensor:
        raise NotImplementedError

@dataclass
class MovingAverage(Weight):
    window_size : float
    decay: Union[float, List[Union[float, str]]]
    channel_weight: Union[float, List[Union[float, str]]] = 1.
    def __post_init__(self):
        if isinstance(self.decay, List):
            self.decay = [float(c_) for c_ in self.decay]
        elif isinstance(self.decay, float):
            self.decay = [self.decay]
        if isinstance(self.channel_weight, List):
            self.channel_weight = [float(a) for a in self.channel_weight]
        elif isinstance(self.channel_weight, float):
            self.channel_weight = [self.channel_weight]
        self.c_ = torch.tensor(self.decay)[None, None, :, None] # bz=1, tok=1, channels
        self.cw = torch.tensor(self.channel_weight)[None, None, :, None] # bz=1, tok=1, channels
    def __call__(self, time_diffs : torch.FloatTensor, metric_vals : torch.FloatTensor) -> torch.Tensor:
        # NOTE: (bz, observations, channels, eval_points)
        # NOTE: numerical stability for 0 * inf
        e = 1e-9
        mov_avg = self.cw*(-self.c_*(time_diffs+e)).exp()
        mov_avg[time_diffs>=self.window_size] = 0
        return mov_avg
@dataclass
class Hann(Weight):
    window_size : float
    channel_weight: Union[float, List[Union[float, str]]] = 1.
    def __post_init__(self):
        if isinstance(self.channel_weight, float):
            self.channel_weight = [self.channel_weight]
        self.cw = torch.tensor(self.channel_weight)[None, None, :, None] # bz=1, tok=1, channels
    def __call__(self, time_diffs : torch.FloatTensor, metric_vals : torch.FloatTensor) -> torch.Tensor:
        # NOTE: (bz, observations, channels, eval_points)
        weight = self.cw*(0.5 +.5*np.cos(np.pi*time_diffs/self.window_size)) # /self.window_size
        weight[time_diffs>=self.window_size] = 0
        return weight


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
    def end_window(self, end_window :int):
        self._end_window = end_window

    @property
    def name(self) -> str:
        return Path(self.path).stem + f'_start_{self.start_node}_nodes_{self.n_inpaint}' + ('' if self.end_window is None else f'_end_{self.end_window}')
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
    inpaint_end : float
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
    logp = torch.nn.functional.log_softmax(logits, dim=dim)
    # NOTE: define that when p==0, then we return 0...
    return -torch.where(p>0 , p * logp, p)

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
@dataclass
class Experiment:
    time_points_generator: TimepointsGenerator
    weight : Weight
    dataset : Data
    ic_curve: Optional[ICCurve]
    match_metric: str = 'ic'
    # metric_clip: Optional[torch.FloatTensor] = None
    metric_clip: Optional[List[float]] = None
    onset_on_next_note: bool = True
    reduce_equal_times: str = 'sum'
    # NOTE: here we should have either the test set, or some named collection of pieces....
    def __post_init__(self):
        assert self.match_metric in ['ic', 'typicality']
        self.metric_clip_ = torch.FloatTensor(self.metric_clip) if self.metric_clip is not None else None
    @property
    def hash_name(self)-> str:
        return hashlib.sha256(str(self).encode('utf-8')).hexdigest()


@dataclass
class SamplingConfig:
    k_traces: int
    temperature : float
    n_poly_notes : Optional[int] = None
    dynamic_temperature_max_ic: Optional[float ] = None
    top_p: float = 0.0
    top_k: int = 0