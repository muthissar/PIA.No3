import torch


def numerial_stable_softmax_entr(logits, dim=-1):
    p = torch.nn.functional.softmax(logits, dim=dim)
    logp = torch.nn.functional.log_softmax(logits, dim=dim)
    # NOTE: define that when p==0, then we return 0...
    return -torch.where(p>0 , p * logp, p)