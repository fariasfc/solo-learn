import argparse
from typing import Any, Dict, List, Sequence, Tuple
from solo.utils.metrics import accuracy_at_k, weighted_mean
from copy import deepcopy, copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumModel
from solo.utils.momentum import initialize_momentum_params

from solo import metrics_utils

class BYOL(BaseMomentumModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        **kwargs,
    ):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Args:
            output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BYOL, BYOL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("byol")

        # projector
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online encoder (encoder, projector and predictor).

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the logits of the head.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # forward momentum encoder
        with torch.no_grad():
            z1_momentum = self.momentum_projector(momentum_feats1)
            z2_momentum = self.momentum_projector(momentum_feats2)

        # ------- contrastive loss -------
        neg_cos_sim = byol_loss_func(p1, z2_momentum) + byol_loss_func(p2, z1_momentum)

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum encoder and classifier, such as forwarding a batch of images in the momentum
        encoder and classifier and computing statistics.

        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """
        batch2 = copy(batch)
        batch[0] = batch[0][0]
        batch2[0] = batch2[0][1]

        
        parent_metrics, metrics = super().validation_step(batch, batch_idx)
        parent_metrics2, metrics2 = super().validation_step(batch2, batch_idx)
        feats1 = parent_metrics['feats']
        feats2 = parent_metrics2['feats']

        # momentum_outs = [self._shared_step_momentum(b[0], b[1]) for b in [batch, batch2]]
        # momentum_feats1, momentum_feats2 = momentum_outs[0]['feats'], momentum_outs[1]['feats']

        with torch.no_grad():
            z1 = self.projector(feats1)
            z2 = self.projector(feats2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)

            # # forward momentum encoder
            # z1_momentum = self.momentum_projector(momentum_feats1)
            # z2_momentum = self.momentum_projector(momentum_feats2)

        X, targets = batch
        batch_size = targets.size(0)

        metrics_utils.align_loss(p1, p2)
        latent_metrics = {
            "our_alignment": metrics_utils.align_loss(p1, p2),
            "our_normalized_alignment": metrics_utils.align_loss(p1, p2, normalized=True),
            "our_cosine_distance": metrics_utils.cos_dist_loss(p1, p2),
            "our_uniformity": metrics_utils.uniform_loss(p1),
            "our_normalized_uniformity": metrics_utils.uniform_loss(p1, normalized=True)
        }
        if metrics is not None:
            metrics = {**metrics, **latent_metrics}
        else:
            metrics = latent_metrics

        metrics["batch_size"] = batch_size

        out = self._shared_step_momentum(X, targets)
        # out = momentum_outs[0]

        if self.momentum_classifier is not None:
            metrics = {
                "momentum_val_loss": out["loss"],
                "momentum_val_acc1": out["acc1"],
                "momentum_val_acc5": out["acc5"],
            }

        return parent_metrics, metrics

    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages the losses and accuracies of the momentum encoder / classifier for all the
        validation batches. This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (Tuple[List[Dict[str, Any]]]):): list of outputs of the validation step for self
                and the parent.
        """

        parent_outs = [out[0] for out in outs]
        super().validation_epoch_end(outs)
        # online_predictions = parent_outs['logits']
        # target_projections = parent_outs['target_projections']
        momentum_outs = [out[1] for out in outs]
        log = {
            "our_alignment": weighted_mean(momentum_outs, "our_alignment", "batch_size"),
            "our_normalized_alignment": weighted_mean(momentum_outs, "our_normalized_alignment", "batch_size"),
            "our_uniformity": weighted_mean(momentum_outs, "our_uniformity", "batch_size"),
            "our_normalized_uniformity": weighted_mean(momentum_outs, "our_normalized_uniformity", "batch_size"),
            "our_cosine_distance": weighted_mean(momentum_outs, "our_cosine_distance", "batch_size"),
        }

        self.log_dict(log, sync_dist=True)

        if self.momentum_classifier is not None:

            val_loss = weighted_mean(momentum_outs, "momentum_val_loss", "batch_size")
            val_acc1 = weighted_mean(momentum_outs, "momentum_val_acc1", "batch_size")
            val_acc5 = weighted_mean(momentum_outs, "momentum_val_acc5", "batch_size")

            log = {
                "momentum_val_loss": val_loss,
                "momentum_val_acc1": val_acc1,
                "momentum_val_acc5": val_acc5,
            }
            self.log_dict(log, sync_dist=True)
