import torch
from torch import nn
import torch.nn.functional as F

def align_loss(x, y, alpha=2, normalized=False, average=True):
    """
    calculate alignment metric from embedding pairs
    """
    if normalized:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

    alignments = (x - y).norm(p=2, dim=1).pow(alpha)

    if average:
        alignments = alignments.mean()

    return alignments


def cos_dist_loss(x, y, average=True):
    """
    calculate alignment metric (cosine distance) from embedding pairs. 0: perfect alignment, 1: perpendicular, 2: opposite
    """
    # cos_similarity = (x * y).sum(-1) / (torch.norm(x, p=2, dim=1) * torch.norm(y, p=2, dim=1))
    cos_similarity = nn.CosineSimilarity()(x, y)

    if average:
        cos_similarity = cos_similarity.mean()

    return 1 - cos_similarity


def uniform_loss(x, t=2, normalized=False, average=True):
    """
    calculate uniformity metric from embeddings (not including augmented counterparts)
    """
    if normalized:
        x = F.normalize(x, p=2, dim=1)

    if average:
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    else:
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().log()