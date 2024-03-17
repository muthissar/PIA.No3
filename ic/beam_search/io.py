from typing import Optional
from typing_extensions import Self
from ic.ic import Piece


import torch


from dataclasses import dataclass
from pathlib import Path


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