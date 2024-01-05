from CIA.ic import Experiment, SamplingConfig


from slugify import slugify


import dataclasses
import logging
from pathlib import Path


@dataclasses.dataclass
class Config:
    sampling_config: SamplingConfig
    samples_per_template: int
    logging: str
    experiment: Experiment
    def __post_init__(self):
        # TODO: exp uniquely identifies where
        args_str =  f'{slugify(str(self.experiment))}/{slugify(str(self.sampling_config))}'
        self.out = Path(f'out/{args_str}')
        self.out.mkdir(parents=True, exist_ok=True)
        numeric_level = getattr(logging, self.logging.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % self.logging)
        log_file = self.out.joinpath('log.txt')
        # TODO: this is problematic, since we need to have one per experiment and since apparantly with 
        # the multiproc we do cannot ask for a nemed one?
        logging.basicConfig(filename=log_file, level=numeric_level)