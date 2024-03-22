from dataclasses import dataclass, field
from warnings import warn
import torch


from typing import List, Tuple


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
        assert round(self.step / self.eval_step, 5).is_integer()
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
    def get_eval_points(self , completed_idx : torch.BoolTensor):
        if self.current_step == 0:
            ts = self.k_traces*[torch.tensor([0.0])]
        else:
            ts = []
            max_time = self.next_time_traces.max()
            for i, next_time_trace in enumerate(self.next_time_traces):
                e = 1e-9
                # NOTE: overshoot next_time_trace - (self.current_step) * self.step
                # NOTE: this is 
                # TODO: inefficient, to check all points, but important, when using sequences that are done.
                if i in completed_idx:
                    end_int = max_time
                    start_int = 0
                else:
                    end_int = next_time_trace-e
                    # start_int = (self.current_step-1) * self.step
                    start_int = 0
                ts_ = torch.arange(start_int, end_int, self.eval_step)
                # if len(ts_) == 0 or ts_[-1] != next_time_trace:
                #     ts_ = torch.cat([ts_, torch.tensor([next_time_trace])])
                ts.append(ts_)

            # ts = torch.arange((self.current_step-1) * self.step, self.current_step * self.step, self.eval_step)
        # return ts[None]
        return ts
    def get_all_eval_points(self):
        return torch.arange(0, self.placeholder_duration, self.eval_step)[None]

# NOTE: does not work do to not finishing time points
# @dataclass
# class SingleNoteTimepoints(TimepointsGenerator):
#     # def __post_init__(self):
#     #     # assert (self.step / self.eval_step).is_integer()
#     #     self.best_times = [0.0]
#     k_traces : int = field(repr=False)
#     eval_step: float = field(repr=False, default=0.1)
#     tol_placeholder_duration : float = field(repr=False, default=0.2)
#     def initialize(self, placholder_duration : float):
#         # self.current_time = 0.0
#         self.placeholder_duration = placholder_duration
#         # self.past_time_traces =  torch.zeros(self.k_traces)
#         # self.current_time_traces =  torch.zeros(self.k_traces)
#         self.next_time_traces = torch.zeros(self.k_traces)
#         self.best_times = [0.0]
#         warn('This value is hardcoded, and the same in decoder_events_handler.py')
#     def update_is_exceeded(self, t : torch.FloatTensor, idx : int) -> bool:
#         '''
#         Times
#         '''
#         # self.current_time_traces[idx] = self.next_time_traces[idx]
#         # TODO: the ics of shift should be set on the next note...
#         self.next_time_traces[idx] = t
#         return True
#     def done(self) -> bool:
#         return len(self.best_times)>0 and self.best_times[-1] >= self.placeholder_duration - self.tol_placeholder_duration
#     def update_step(self, idx : int) -> None:
#         # self.best_times.append(t.item())
#         # raise NotImplementedError('This does not work if generation[idx] is done... ')
#         self.best_times.append(self.next_time_traces[idx].item())
#         # self.time_traces = []
#     def progress(self) -> Tuple[int, int]:
#         '''
#         Returns the (rounded) current number of secs generated and the total number of secs to be generated
#         '''
#         return round(self.best_times[-1] if len(self.best_times) else 0.), round(self.placeholder_duration)
#     def get_eval_points(self) -> List[torch.FloatTensor]:
#         # if len(self.best_times) == 0:
#         #     ts = torch.zeros(self.k_traces, 1)
#         # else:
#         #     ts = torch.tensor(self.time_traces)[:, None]
#         # ts = self.current_time_traces[:, None]
#         # ts = torch.tensor(self.best_times[-1])[None, None]
#         # NOTE: quite inefficient, because in principle it would be enough to only use 
#         # next_time_traces, because we always expand from the note before. However, for times where the sequence is done, 
#         # we need to evaluate in the extra points. Can  we simply get rid of choosing the done sequences?
#         # ts = torch.cat([torch.tensor(self.best_times)[None].expand(self.k_traces, -1), self.next_time_traces[:, None]], dim=1)
#         # ts = self.next_time_traces[:, None]
#         ts = []
#         for next_time_trace in self.next_time_traces:
#             ts_ = torch.arange(self.best_times[-1], next_time_trace, self.eval_step)
#             if len(ts_) == 0 or ts_[-1] != next_time_trace:
#                 ts_ = torch.cat([ts_, torch.tensor([next_time_trace])])
#             ts.append(ts_)
#         return ts
#     def get_all_eval_points(self):
#         # return torch.arange(0, self.placeholder_duration, self.eval_step)
#         return torch.tensor(self.best_times)[None]