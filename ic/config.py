from dataclasses import dataclass
import hashlib
from typing import List, Optional


from slugify import slugify


import dataclasses
import logging
from pathlib import Path

import torch

from ic.curves import ICCurve
from ic.curves.weight import Weight
from ic.data import Data
from ic.beam_search.timepoints import TimepointsGenerator


@dataclass
class SamplingConfig:
    k_traces: int
    temperature : float
    n_poly_notes : Optional[int] = None
    dynamic_temperature_max_ic: Optional[float ] = None
    top_p: float = 0.0
    top_k: int = 0


@dataclass
class Experiment:
    time_points_generator: TimepointsGenerator
    weight : Weight
    dataset : Optional[Data]
    ic_curve: Optional[ICCurve]
    match_metric: str = 'ic'
    # metric_clip: Optional[torch.FloatTensor] = None
    metric_clip: Optional[List[float]] = None
    onset_on_next_note: bool = True
    reduce_equal_times: str = 'sum'
    receptive_field_len: Optional[int] = None
    

    # NOTE: here we should have either the test set, or some named collection of pieces....
    def __post_init__(self):
        assert self.match_metric in ['ic', 'typicality']
        self.metric_clip_ = torch.FloatTensor(self.metric_clip) if self.metric_clip is not None else None
    @property
    def hash_name(self)-> str:
        return hashlib.sha256(str(self).encode('utf-8')).hexdigest()

@dataclasses.dataclass
class BaseConfig:
    sampling_config: SamplingConfig
    experiment: Experiment

@dataclasses.dataclass
class Config(BaseConfig):
    samples_per_template: int
    logging: str
    seed: Optional[int] = None
    def __post_init__(self):
        # TODO: exp uniquely identifies where
        self.out.mkdir(parents=True, exist_ok=True)
        numeric_level = getattr(logging, self.logging.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % self.logging)
        log_file = self.out.joinpath('log.txt')
        # TODO: this is problematic, since we need to have one per experiment and since apparantly with 
        # the multiproc we do cannot ask for a nemed one?
        logging.basicConfig(filename=log_file, filemode='a', level=numeric_level)
    @property
    def out(self) -> Path:
        exp_folder = self.experiment.hash_name
        # exp_folder = slugify(str(self.experiment))
        args_str =  f'{exp_folder}/{slugify(str(self.sampling_config))}'
        return Path(f'out/{args_str}')
