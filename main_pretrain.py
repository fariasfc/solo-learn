import os
import sys
sys.path.append('/datadrive/solo-learn/')
from pprint import pprint
from torch import nn
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from solo.args.setup import parse_args_pretrain
from solo.methods import METHODS

try:
    from solo.methods.dali import PretrainABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

from solo.utils.checkpointer import Checkpointer
from solo.utils.classification_dataloader import prepare_data as prepare_data_classification
from solo.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
)

class LatentMetricsCallback(Callback):
    def __init__(self, wandb_logger) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger

    def align_loss(x, y, alpha=2, normalized=False):
        """
        calculate alignment metric from embedding pairs
        """
        if normalized:
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(y, p=2, dim=1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()


    def cos_dist_loss(x, y):
        """
        calculate alignment metric (cosine distance) from embedding pairs. 0: perfect alignment, 1: perpendicular, 2: opposite
        """
        # cos_similarity = (x * y).sum(-1) / (torch.norm(x, p=2, dim=1) * torch.norm(y, p=2, dim=1))
        cos_similarity = nn.CosineSimilarity()(x, y).mean()
        return 1 - cos_similarity

    def uniform_loss(x, t=2, normalized=False):
        """
        calculate uniformity metric from embeddings (not including augmented counterparts)
        """
        if normalized:
            x = F.normalize(x, p=2, dim=1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    # def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     metrics = {
    #         "align":
    #     }
    #     self.wandb_logger.log_metrics()
    #     return super().on_validation_epoch_end(trainer, pl_module)


def main():
    seed_everything(5)

    args = parse_args_pretrain()

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    MethodClass = METHODS[args.method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = type(f"Dali{MethodClass.__name__}", (MethodClass, PretrainABC), {})

    model = MethodClass(**args.__dict__)

    # contrastive dataloader
    if not args.dali:
        # asymmetric augmentations
        if args.unique_augs > 1:
            transform = [
                prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
                for kwargs in args.transform_kwargs
            ]
        else:
            transform = prepare_transform(
                args.dataset, multicrop=args.multicrop, **args.transform_kwargs
            )

        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        if args.multicrop:
            assert not args.unique_augs == 1

            if args.dataset in ["cifar10", "cifar100"]:
                size_crops = [32, 24]
            elif args.dataset == "stl10":
                size_crops = [96, 58]
            # imagenet or custom dataset
            else:
                size_crops = [224, 96]

            transform = prepare_multicrop_transform(
                transform, size_crops=size_crops, num_crops=[args.num_crops, args.num_small_crops]
            )
        else:
            if args.num_crops != 2:
                assert args.method == "wmse"

            transform = prepare_n_crop_transform(transform, num_crops=args.num_crops)

        train_dataset = prepare_datasets(
            args.dataset,
            transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            no_labels=args.no_labels,
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # normal dataloader for when it is available
    if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
        val_loader = None
    elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
        val_loader = None
    else:
        _, val_loader = prepare_data_classification(
            args.dataset,
            transform=transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.method),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

        if args.auto_umap:
            assert (
                _umap_available
            ), "UMAP is not currently avaiable, please install it first with [umap]."
            auto_umap = AutoUMAP(
                args,
                logdir=os.path.join(args.auto_umap_dir, args.method),
                frequency=args.auto_umap_frequency,
            )
            callbacks.append(auto_umap)

        # callbacks.append(LatentMetricsCallback(wandb_logger))

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp",
    )

    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
