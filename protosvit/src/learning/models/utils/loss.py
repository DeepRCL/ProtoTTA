"""
From https://github.com/stevenstalder/NN-Explainer/blob/main/src/utils/loss.py
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchmetrics.functional import pairwise_cosine_similarity


def norm(t):
    return F.normalize(t, dim=-1, eps=1e-10)


def tensor_correlation(a, b):
    return torch.einsum("nhc,nic->nhi", a, b)


class OrthoLoss(nn.Module):
    """
    Orthogonal Loss module.

    This module computes the orthogonal loss, which encourages the prototype matrix to be orthogonal.

    Args:
        nb_prototypes (int): The number of prototypes.

    Returns:
        torch.Tensor: The computed loss.

    """

    def __init__(self, nb_prototypes: int):
        super().__init__()
        self.eye = nn.Parameter(torch.eye(nb_prototypes), requires_grad=False)

    def forward(self, matrix, return_diff=False):
        # Normalize the rows
        matrix = matrix / (matrix.norm(dim=1, keepdim=True) + 1e-8)
        # Compute the product of the matrix and its transpose
        product = torch.mm(matrix, matrix.t())

        # Compute the difference between the product and the identity matrix
        diff = product - self.eye.to(matrix.device)
        diff_triu = diff.triu(diagonal=1)

        # Compute the loss as the square root of the sum of squares of the differences
        loss = torch.sqrt(torch.sum(diff_triu**2))
        if return_diff:
            return loss, diff
        else:
            return loss


class WeightedHsLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        weight_loss: float = 1.0,
        structured: bool = True,  # grouped or ungrouped HS LOSS
        compute_importance: bool = True,  # whether loss appplied on loss or importance
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight_loss = weight_loss
        self.structured = structured
        self.compute_importance = compute_importance

    def forward(
        self,
        classification_head_weight: torch.Tensor,
        similarity_score: torch.Tensor = None,
        eps: float = 1e-20,
    ) -> torch.Tensor:
        if self.compute_importance:
            importance = torch.einsum(
                "bp,cp->bpc", similarity_score, classification_head_weight
            )
            if self.structured:
                sparsity_loss = (
                    self.alpha
                    * (torch.sum(torch.norm(importance, p=2, dim=1), dim=1) ** 2)
                    + self.beta
                    * (torch.sum(torch.norm(importance, p=2, dim=2), dim=1) ** 2)
                ) / (torch.norm(importance, p=2, dim=[1, 2]) ** 2 + 1e-20)
                sparsity_loss = sparsity_loss / torch.numel(importance[0]) ** 0.5
                sparsity_loss += self.gamma * torch.norm(importance, dim=[1, 2], p=2)
                sparsity_loss = sparsity_loss.mean()

            else:
                sparsity_loss = self.alpha * (
                    torch.norm(importance, p=1, dim=[1, 2]) ** 2
                    / torch.sum(importance**2 + eps, dim=[1, 2])
                )
                sparsity_loss = sparsity_loss / torch.numel(importance[0]) ** 0.5
                sparsity_loss += self.gamma * torch.norm(importance, dim=[1, 2], p=2)

                sparsity_loss = sparsity_loss.mean()

        else:
            # classification head weight dim are inversed so inverse beta and alpha
            if self.structured:
                sparsity_loss = (
                    self.beta
                    * (
                        torch.sum(
                            torch.norm(classification_head_weight, p=2, dim=0), dim=0
                        )
                        ** 2
                    )
                    + self.alpha
                    * (
                        torch.sum(
                            torch.norm(classification_head_weight, p=2, dim=1), dim=0
                        )
                        ** 2
                    )
                ) / (torch.norm(classification_head_weight, p=2) ** 2 + 1e-20)
                sparsity_loss = (
                    sparsity_loss / torch.numel(classification_head_weight) ** 0.5
                )
                sparsity_loss += self.gamma * torch.norm(
                    classification_head_weight, p=2
                )

            else:
                sparsity_loss = self.alpha * (
                    torch.norm(classification_head_weight, p=1) ** 2
                    / torch.sum(classification_head_weight**2 + eps)
                )
                sparsity_loss = (
                    sparsity_loss / torch.numel(classification_head_weight) ** 0.5
                )
                sparsity_loss += self.gamma * torch.norm(
                    classification_head_weight, p=2
                )

        return self.weight_loss * sparsity_loss


class HsLoss(nn.Module):
    def __init__(
        self,
        weight_loss: float = 1.0,
    ):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        sparsity_loss = (
            torch.norm(input + 1e-20, dim=[1, 2], p=1) ** 2
            / torch.sum(input + 1e-20**2, dim=[1, 2])
        ).mean()
        sparsity_loss = sparsity_loss / torch.numel(input[0]) ** 0.5

        return self.weight_loss * sparsity_loss


class L1Loss(nn.Module):
    def __init__(self, weight_loss: float = 1.0, **kwargs):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self, classification_head_weight) -> torch.Tensor:
        sparsity_loss = torch.norm(classification_head_weight, p=1)
        sparsity_loss = sparsity_loss / torch.numel(classification_head_weight) ** 0.5

        return self.weight_loss * sparsity_loss


class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        loss_version: int = 1,
        reweighting: int = 1,
        weight_loss: float = 1.0,
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.loss_version = loss_version
        self.reweighting = reweighting
        self.weight_loss = weight_loss

    def forward(
        self,
        modeloutput_z: torch.Tensor,
        prototypes: torch.Tensor,
        rho: float = 0.02,
    ):
        device = torch.device("cuda") if modeloutput_z.is_cuda else torch.device("cpu")
        prototypes = F.normalize(prototypes.float(), dim=-1)
        modeloutput_z = F.normalize(modeloutput_z, dim=-1)

        spatial_size = modeloutput_z.shape[1]
        modeloutput_z = modeloutput_z.reshape(-1, modeloutput_z.shape[-1])
        batch_size = modeloutput_z.shape[0]
        split = int(spatial_size)
        mini_iters = int(batch_size / split)

        negative_mask_one = torch.scatter(
            torch.ones(
                (split, batch_size),
            ),
            1,
            torch.arange(split).view(-1, 1),
            0,
        ).to(device)
        mask_neglect_base = torch.FloatTensor(split, batch_size).uniform_() < rho
        mask_neglect_base = mask_neglect_base.cuda()

        loss = torch.tensor(0).to(device)
        Rpoint = torch.matmul(modeloutput_z, prototypes.transpose(0, 1))

        Rpoint = torch.max(Rpoint, dim=1).values
        Rpoint_T = Rpoint.unsqueeze(-1).repeat(1, split)

        for mi in range(mini_iters):
            modeloutput_z_one = modeloutput_z[mi * split : (mi + 1) * split]
            output_cossim_one = torch.matmul(
                modeloutput_z_one, modeloutput_z.transpose(0, 1)
            )

            output_cossim_one_T = output_cossim_one.transpose(0, 1)
            mask_one_T = Rpoint_T < output_cossim_one_T
            mask_one_T = torch.tensor(mask_one_T.transpose(0, 1))
            Rpoint_one = Rpoint[mi * split : (mi + 1) * split]
            Rpoint_one = Rpoint_one.unsqueeze(-1).repeat(1, batch_size)
            mask_one = torch.tensor(
                (Rpoint_one < output_cossim_one),
            )
            mask_one = torch.logical_or(mask_one, mask_one_T)
            neglect_mask = torch.logical_or(mask_one, mask_neglect_base)
            neglect_negative_mask_one = negative_mask_one * neglect_mask
            mask_one = mask_one * negative_mask_one

            modeloutput_z_one = modeloutput_z[mi * split : (mi + 1) * split]

            anchor_dot_contrast_one = torch.div(
                torch.matmul(modeloutput_z_one, modeloutput_z.T), self.temperature
            )

            logits_max_one, _ = torch.max(anchor_dot_contrast_one, dim=1, keepdim=True)
            logits_one = anchor_dot_contrast_one - logits_max_one.detach()
            exp_logits_one = torch.exp(logits_one) * neglect_negative_mask_one
            log_prob_one = logits_one - torch.log(exp_logits_one.sum(1, keepdim=True))

            if self.loss_version == 1:
                nonzero_idx = torch.where(mask_one.sum(1) != 0.0)
                mask_one = mask_one[nonzero_idx]
                log_prob_one = log_prob_one[nonzero_idx]
                # mask_ema_one = mask_ema_one[nonzero_idx]
                weighted_mask = mask_one.detach()
                if self.reweighting == 1:
                    pnm = torch.tensor(
                        torch.sum(weighted_mask, dim=1), dtype=torch.float32
                    )
                    pnm = pnm / torch.sum(pnm)
                    pnm = pnm / torch.mean(pnm)
                else:
                    pnm = 1
                mean_log_prob_pos_one = (weighted_mask * log_prob_one).sum(1) / (
                    weighted_mask.sum(1)
                )
                loss = loss - torch.mean(
                    (self.temperature / self.base_temperature)
                    * mean_log_prob_pos_one
                    * pnm
                )

            elif self.loss_version == 2:
                nonzero_idx = torch.where(mask_one.sum(1) != 0.0)
                mask_one = mask_one[nonzero_idx]
                if self.reweighting == 1:
                    pnm = torch.tensor(torch.sum(mask_one, dim=1), dtype=torch.float32)
                    pnm = pnm / torch.sum(pnm)
                    pnm = pnm / torch.mean(pnm)
                else:
                    pnm = 1
                mean_log_prob_pos_one = (mask_one * log_prob_one[nonzero_idx]).sum(
                    1
                ) / (mask_one.sum(1))
                loss = loss - torch.mean(
                    (self.temperature / self.base_temperature)
                    * mean_log_prob_pos_one
                    * pnm
                )

        return loss / mini_iters * self.weight_loss


class WeightedOrthoLoss(nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self, prototypes: torch.Tensor, weight_class: torch.Tensor):
        ortho_loss_proto = torch.matmul(
            prototypes, prototypes.transpose(0, 1)
        ) - torch.eye(prototypes.shape[0], device=prototypes.device)
        weighted_ortho = weight_class.unsqueeze(-1) * ortho_loss_proto.unsqueeze(0)
        ortho_loss_proto = torch.norm(weighted_ortho, dim=[1, 2], p=2).mean()

        return self.weight_loss * ortho_loss_proto


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self, similarity: torch.Tensor):
        # Subtract max similarity from all other similaritie
        max_similarity, idx_max = similarity.max(dim=-1, keepdim=True)
        contrastive_similarity = similarity - max_similarity

        # Apply softmax to get probability distribution over prototypes
        prob = F.softmax(contrastive_similarity, dim=-1)
        test_prob = rearrange(prob, "b c p -> (b c) p")
        # Compute negative log likelihood of correct prototype
        loss = F.nll_loss(test_prob.log(), idx_max.reshape(-1))

        return self.weight_loss * loss


class ConsistencyLoss(torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()
        self.weight_loss = weight_loss
        self.fn = nn.PairwiseDistance()

    def forward(self, embeddings: torch.Tensor, embeddings_aug: torch.Tensor):
        # Subtract max similarity from all other similaritie
        loss_consistency = torch.mean(self.fn(embeddings, embeddings_aug))

        return self.weight_loss * loss_consistency
