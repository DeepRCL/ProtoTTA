import math
from typing import Optional as _Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from einops import rearrange
from matplotlib import pyplot as plt
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.nn import init
from torchmetrics import Accuracy, MetricCollection, Precision, Recall

from src.learning.models.utils.custom_scheduler import CustomLRScheduler
from src.learning.models.utils.dropblock import DropBlock2D
from src.learning.models.utils.gumbell_softmax import gumbel_softmax
from src.learning.models.utils.loss import (
    HsLoss,
    L1Loss,
    WeightedHsLoss,
    WeightedOrthoLoss,
)
from src.learning.models.utils.modules import LayerNorm, NonNegLinear
from src.learning.models.utils.prune_model import FisherPruningHook
from src.learning.models.utils.visualisation import (
    plot_similarity,
    plot_weight_distribution,
    plot_weight_heatmap,
)

torch.set_float32_matmul_precision("medium")


class ClassificationModulePrototype(pl.LightningModule):
    """
    Classification module  with prototype implementation.

    Args:
        nb_prototypes (int): Number of prototypes.
        ortho_loss_fn (function): Orthogonal loss function.
        prototype_dim (int): Dimension of the prototypes.
        image_encoder (nn.Module): Image encoder model.
        num_classes (int): Number of classes.
        lr (float): Learning rate.
        warmup_epochs (int): Number of warmup epochs.
        loss_type (str, optional): Type of loss function. Defaults to "cross_entropy".
        freeze_backbone (bool, optional): Whether to freeze the backbone. Defaults to False.
        decay_factor (float, optional): Decay factor for learning rate if backbone is not freezed. Defaults to 0.5.
        weight_class (Optional[torch.Tensor], optional): Weight for each class. Defaults to None.
        data_mean (List[float], optional): Mean values of the data. Defaults to [0, 0, 0].
        data_std (List[float], optional): Standard deviation values of the data. Defaults to [1, 1, 1].
        max_epochs (int, optional): Maximum number of epochs. Defaults to 400.
        aggregate_similarity (str, optional): Method for aggregating similarity. Defaults to "nonlinear".
        embed_projection (bool, optional): Whether to use embedding projection. Defaults to True.
        bias_classification_head (bool, optional): Whether to use bias in the classification head. Defaults to False.
        threshold_importance (Optional[float], optional): Threshold for importance in classification head. Defaults to 0.0.
        proto_retain (Optional[float], optional): Retain ratio for prototypes. Defaults to None.
    """

    def __init__(
        self,
        nb_prototypes,
        prototype_dim,
        image_encoder,
        num_classes,
        lr,
        warmup_epochs,
        pruning_start,
        loss_type="cross_entropy",
        prune_model_interval=5,
        freeze_backbone=False,
        bacbone_lr_multiple=0.001,
        epsilon=1e-4,
        weight_class=None,
        data_mean=[0, 0, 0],
        data_std=[1, 1, 1],
        max_epochs=400,
        aggregate_similarity="nonlinear",
        embed_projection=True,
        bias_classification_head=False,
        threshold_importance: _Optional[float] = 0.0,
        sparsity_prototype_loss: nn.Module | None = None,
        sparsity_similarity_loss: nn.Module | None = None,
        consistency_loss: nn.Module | None = None,
        dropblock: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.patch_size = self.image_encoder.patch_size
        self.prototype_dim = prototype_dim
        self.freeze_backbone = freeze_backbone
        self.bacbone_lr_multiple = bacbone_lr_multiple
        self.nb_prototypes = nb_prototypes
        self.epsilon = epsilon
        self.lr = lr
        self.prune_model_interval = prune_model_interval
        self.warmup_epochs = warmup_epochs
        self.pruning_start = pruning_start
        self.max_epochs = max_epochs
        self.aggregate_similarity = aggregate_similarity
        self.data_mean = data_mean
        self.data_std = data_std
        self.loss_type = loss_type
        self.prototype_embeddings = nn.Embedding(nb_prototypes, self.prototype_dim)
        self.embed_projection = embed_projection
        # self.temperature = nn.Parameter(torch.tensor(0.2))
        self.temperature = torch.tensor(0.1)
        self.threshold_importance = threshold_importance
        nn.init.orthogonal_(self.prototype_embeddings.weight)
        self.save_hyperparameters(logger=False)
        self.dropout = nn.Dropout2d(0.1)
        self.mask_proto = nn.Parameter(torch.ones(nb_prototypes), requires_grad=False)
        self.sparsity_similarity_loss = sparsity_similarity_loss
        self.sparsity_prototype_loss = sparsity_prototype_loss
        self.consistency_loss = consistency_loss

        self.dropblock = dropblock
        if self.dropblock is not None and self.consistency_loss is not None:
            raise ValueError(
                "Dropblock and consistency loss cannot be used together yet"
            )

        if self.embed_projection:
            self.project_head = nn.Linear(self.prototype_dim, self.prototype_dim)

        if self.aggregate_similarity == "nonlinear":
            self.conv1 = nn.Conv2d(
                self.nb_prototypes,
                self.nb_prototypes,
                (3, 3),
                padding=1,
                groups=self.nb_prototypes,
            )
            self.conv2 = nn.Conv2d(
                self.nb_prototypes,
                self.nb_prototypes,
                (1, 1),
                groups=self.nb_prototypes,
            )
            self.layernorm = LayerNorm(self.nb_prototypes)

        self.classification_head = NonNegLinear(
            nb_prototypes,
            num_classes,
            bias=bias_classification_head,
        )

        if self.loss_type == "binary_cross_entropy":
            metric_collection = MetricCollection(
                [
                    Precision(task="multilabel", num_labels=num_classes),
                    Recall(task="multilabel", num_labels=num_classes),
                ]
            )

            self.classification_loss_fn = F.binary_cross_entropy_with_logits

        elif self.loss_type == "cross_entropy":
            metric_collection = MetricCollection(
                [Accuracy(task="multiclass", num_classes=num_classes)]
            )
            if weight_class is not None:
                self.weight_class = nn.Parameter(
                    torch.tensor(weight_class).float(), requires_grad=False
                )
            else:
                self.weight_class = None
            self.classification_loss_fn = F.cross_entropy
        self.train_metrics = metric_collection.clone(prefix="train/")
        self.valid_metrics = metric_collection.clone(prefix="val/")
        self.test_metrics = metric_collection.clone(prefix="test/")

    def backbone_project(self, x: torch.Tensor):
        model_output = self.image_encoder(x)
        model_output = model_output[1]

        if self.dropblock is not None:
            model_output = self.dropblock(model_output)

        modeloutput_s = model_output.permute(0, 2, 3, 1).reshape(-1, self.prototype_dim)

        if self.embed_projection:
            modeloutput_s = F.relu(modeloutput_s)
            modeloutput_z = self.project_head(modeloutput_s)
            # modeloutput_z = F.relu(modeloutput_z)
        else:
            modeloutput_z = modeloutput_s

        modeloutput_z = modeloutput_z.reshape(x.shape[0], -1, self.prototype_dim)
        modeloutput_z = F.normalize(modeloutput_z, dim=-1)

        return modeloutput_z

    def forward(self, x: torch.Tensor, train=False):
        modeloutput_z = self.backbone_project(x)

        similarity = torch.einsum(
            "bic, bnc -> bni",
            F.normalize(modeloutput_z.float(), dim=-1),
            F.normalize(
                self.prototype_embeddings.weight.unsqueeze(0).repeat(x.shape[0], 1, 1),
                dim=-1,
            ),
        )

        similarity = F.softmax(similarity / (self.temperature), dim=1)
        similarity_prototypes = similarity * self.mask_proto[None, :, None]
        similarity_background = similarity * (1 - self.mask_proto[None, :, None])
        # Keep a pre-head prototype score for TTA filtering. This stays on the
        # original prototype-similarity scale, unlike similarity_score below.
        proto_filter_score = similarity_prototypes.amax(dim=-1)

        similarity_ratio = similarity_background.sum() / similarity.sum()
        self.log("similarity_ratio", similarity_ratio)
        # similarity_prototypes = similarity

        if self.aggregate_similarity == "nonlinear":
            h = w = int(similarity.shape[-1] ** 0.5)
            similarity_reshaped = rearrange(
                similarity_prototypes, "b n (h w) -> b n h w", h=h, w=w
            )
            similarity_1 = self.conv1(similarity_reshaped)
            similarity_2 = self.conv2(similarity_reshaped)
            similarity_reshaped = similarity_1 + similarity_2
            similarity_reshaped = similarity_reshaped.permute(0, 2, 3, 1)
            similarity_reshaped = self.layernorm(similarity_reshaped)
            similarity_reshaped = similarity_reshaped.permute(0, 3, 1, 2)

            similarity_reshaped = F.relu(similarity_reshaped)

            similarity_score = similarity_reshaped.amax(dim=(2, 3))
        else:
            similarity_score = proto_filter_score

        importance, logits = self.classification_head(
            similarity_score,
            threshold=self.threshold_importance,
            train=train,
        )

        total_out_dict = {
            "projected_proto": self.prototype_embeddings.weight,
            "pred": logits,
            "similarity_prototype": similarity_prototypes,
            "similarity_background": similarity_background,
            "similarity_score": similarity_score,
            "proto_filter_score": proto_filter_score,
            "importance": importance,
            "modeloutput_z": modeloutput_z,
        }

        return total_out_dict

    def configure_optimizers(self):
        if self.freeze_backbone:
            for p in self.image_encoder.model.parameters():
                p.requires_grad = False
            self.image_encoder.model.eval()
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        else:
            param_groups = []
            for i, layer in enumerate(self.image_encoder.model.blocks):
                if i < len(self.image_encoder.model.blocks) - 1:
                    layer.eval()
                    for p in layer.parameters():
                        p.requires_grad = False
                else:
                    lr = self.lr * (self.bacbone_lr_multiple)
                    param_groups.append({"params": layer.parameters(), "lr": lr})
            for param in self.named_parameters():
                if "image_encoder.model." not in param[0]:
                    param_groups.append({"params": param[1], "lr": self.lr})

            optimizer = optim.AdamW(param_groups)

        scheduler_custom = CustomLRScheduler(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            min_lr=1e-4,
            max_epochs=self.max_epochs,
        )
        return [optimizer], [scheduler_custom]

    def _calculate_loss(self, batch, mode="train", plot_similarity=False):
        train = True if mode == "train" else False

        if len(batch) == 3:
            imgs, labels, img_aug = batch
            out_dict = self.forward(imgs, train)
            modeloutput_z_aug = self.backbone_project(img_aug)
            if self.consistency_loss is not None:
                loss_consistency = self.consistency_loss(
                    out_dict["modeloutput_z"], modeloutput_z_aug
                )
            else:
                loss_consistency = torch.tensor(0, device=self.device)
        else:
            imgs, labels = batch
            out_dict = self.forward(imgs, train)
            loss_consistency = torch.tensor(0, device=self.device)

        preds = out_dict["pred"]
        if self.loss_type == "binary_cross_entropy":
            labels = labels.float()
            loss_classification = self.classification_loss_fn(preds, labels)
        elif self.loss_type == "cross_entropy":
            loss_classification = self.classification_loss_fn(
                preds, labels, weight=self.weight_class
            )
        self.log(
            f"{mode}/loss_classification",
            loss_classification,
        )
        loss = loss_classification
        similarity = (
            out_dict["similarity_prototype"] + out_dict["similarity_background"]
        )
        similarity_tmp = rearrange(similarity, "b p n -> (b n) p")
        l_t = -(torch.log(torch.tanh(torch.sum(similarity_tmp, dim=0)) + 1e-20).mean())
        self.log(f"{mode}/loss_t", l_t)
        loss += l_t
        loss += (
            torch.clip((torch.tensor(self.current_epoch / self.warmup_epochs)), 0, 1)
            * loss_consistency
        )

        self.log(
            f"{mode}/loss_consistency",
            loss_consistency,
        )

        if isinstance(self.sparsity_prototype_loss, WeightedHsLoss):
            similarity_score = out_dict["similarity_score"]
            sparsity_prototype_loss = self.sparsity_prototype_loss(
                self.classification_head.weight * self.mask_proto[None, :],
                similarity_score,
            )
        elif isinstance(self.sparsity_prototype_loss, L1Loss):
            sparsity_prototype_loss = self.sparsity_prototype_loss(
                self.classification_head.weight
            )
        else:
            sparsity_prototype_loss = torch.tensor(0, device=self.device)

        # if self.current_epoch > self.warmup_epochs:
        loss += (
            torch.clip((torch.tensor(self.current_epoch / self.warmup_epochs)), 0, 1)
            * sparsity_prototype_loss
        )
        self.log(f"{mode}/sparsity_prototype_loss", sparsity_prototype_loss)

        if isinstance(self.sparsity_similarity_loss, HsLoss):
            similarity_prototype = out_dict["similarity_prototype"]
            sparsity_similarity_loss = self.sparsity_similarity_loss(
                similarity_prototype
            )

            if self.current_epoch > self.warmup_epochs:
                loss += sparsity_similarity_loss
            self.log(f"{mode}/sparsity_similarity_loss", sparsity_similarity_loss)
        else:
            sparsity_similarity_loss = torch.tensor(0, device=self.device)

        self.log("%s/loss" % mode, loss)

        if mode == "train":
            metrics = self.train_metrics.update(preds, labels)

        elif mode == "val":
            metrics = self.valid_metrics.update(preds, labels)
        elif mode == "test":
            metrics = self.test_metrics.update(preds, labels)

        if plot_similarity:
            self.plot_signal(imgs, out_dict)

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._calculate_loss(batch, mode="train")
        return loss

    def on_train_epoch_start(self):
        self.fisher_hook = FisherPruningHook(self.classification_head)

    def on_train_epoch_end(self):
        output = self.train_metrics.compute()
        self.log_dict(output)
        self.train_metrics.reset()

        fig = self.fisher_hook.plot_fisher()
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.logger.experiment.add_figure(f"plot_gradient", fig, self.global_step)
        elif isinstance(self.logger, pl.loggers.WandbLogger):
            wandb.log({f"plot_gradient": wandb.Image(fig)})
        if self.prune_model_interval is not None:
            if (self.current_epoch % self.prune_model_interval == 0) & (
                self.current_epoch >= self.pruning_start
            ):
                prune_mask = self.fisher_hook.compute_prune_mask(self.mask_proto.data)
                # self.mask_proto.data = self.mask_proto.data * prune_mask
                self.mask_proto.data = prune_mask
            nb_proto_remaining = self.mask_proto.sum()
            self.log("Remaining prototypes left", nb_proto_remaining)

        self.fisher_hook.reset()

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.valid_metrics.reset()

    def validation_step(self, batch, batch_idx):
        plot_similarity = True if batch_idx == 0 else False
        self._calculate_loss(batch, mode="val", plot_similarity=plot_similarity)

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.test_metrics.reset()

    @rank_zero_only
    def plot_signal(self, imgs, pred_dict):
        fig1 = plot_similarity(
            imgs,
            pred_dict,
            torch.tensor(self.data_mean),
            torch.tensor(self.data_std),
            fig_nb=3,
            nb_proto_plot=5,
        )
        fig2 = plot_weight_distribution(
            self.classification_head.weight.detach().cpu().numpy()
        )
        fig3 = plot_weight_heatmap(
            (self.classification_head.weight * self.mask_proto[None, :])
            .detach()
            .cpu()
            .numpy()
        )
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.logger.experiment.add_figure(
                f"plot_similarity", fig1, self.global_step
            )
            self.logger.experiment.add_figure(
                f"plot_weight_distribution", fig2, self.global_step
            )
            self.logger.experiment.add_figure(
                f"plot_weight_heatmap", fig3, self.global_step
            )
        elif isinstance(self.logger, pl.loggers.WandbLogger):
            wandb.log({f"Sample": fig1})
            wandb.log({f"weight_distribution": wandb.Image(fig2)})
            wandb.log({f"weight_heatmap": wandb.Image(fig3)})

        plt.close("all")
