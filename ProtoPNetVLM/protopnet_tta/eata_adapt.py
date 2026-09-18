"""
EATA: Efficient Anti-forgetting Test-Time Adaptation.

Adapted from ProtoViT TTA implementation for ProtoPNet.
Based on: https://arxiv.org/abs/2204.02610
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .tent import collect_params, configure_model, copy_model_and_optimizer, softmax_entropy


class EATA(nn.Module):
    """EATA adapts a model by entropy minimization during testing.

    EATA improves upon Tent by:
    1. Filtering unreliable samples (high entropy)
    2. Filtering redundant samples (similar to seen samples)
    3. Optional Fisher regularization to prevent forgetting
    """
    def __init__(self, model, optimizer, fishers=None, fisher_alpha=2000.0,
                 steps=1, episodic=False, e_margin=None, d_margin=0.05,
                 num_classes=5):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.num_samples_update_1 = 0  # after First filtering (unreliable)
        self.num_samples_update_2 = 0  # after Second filtering (redundant)

        # E_0: entropy threshold (Eqn. 3) - default is 0.4 * ln(num_classes)
        if e_margin is None:
            e_margin = math.log(num_classes) * 0.4
        self.e_margin = e_margin
        self.d_margin = d_margin  # cosine similarity threshold (Eqn. 5)

        self.current_model_probs = None  # moving average of probability vector (Eqn. 4)

        self.fishers = fishers  # fisher regularizer for anti-forgetting (Eqn. 9)
        self.fisher_alpha = fisher_alpha  # trade-off beta (Eqn. 8)

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # Adaptation tracking
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'total_updates': 0,
        }

    def forward(self, x):
        if self.episodic:
            self.reset()

        batch_size = x.size(0)
        self.adaptation_stats['total_samples'] += batch_size

        outputs = None

        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs = \
                    forward_and_adapt_eata(x, self.model, self.optimizer, self.fishers,
                                          self.e_margin, self.current_model_probs,
                                          fisher_alpha=self.fisher_alpha,
                                          d_margin=self.d_margin)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)

                self.adaptation_stats['adapted_samples'] += num_counts_2
                if num_counts_2 > 0:
                    self.adaptation_stats['total_updates'] += num_counts_2
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state, strict=True)
        self.current_model_probs = None

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    def forward_no_adapt(self, x):
        """Forward pass without adaptation."""
        return self.model(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def update_model_probs(current_model_probs, new_probs):
    """Update the moving average of model probabilities."""
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


@torch.enable_grad()
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin,
                           current_model_probs, fisher_alpha=2000.0, d_margin=0.05):
    """Forward and adapt model on batch of data with EATA filtering."""
    # Forward - handle ProtoPNet output format
    outputs = model(x)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    # If no optimizer, just return (no adaptation possible)
    if optimizer is None:
        return outputs, 0, 0, current_model_probs

    # Compute entropy
    entropys = softmax_entropy(logits)

    # Filter 1: Remove unreliable samples (high entropy)
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0] > -0.1)
    entropys = entropys[filter_ids_1]

    # Filter 2: Remove redundant samples (similar to seen samples)
    if current_model_probs is not None:
        probs = logits.softmax(1)
        if filter_ids_1[0].size(0) > 0:
            cosine_similarities = F.cosine_similarity(
                current_model_probs.unsqueeze(dim=0),
                probs[filter_ids_1], dim=1
            )
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(current_model_probs, probs[filter_ids_1][filter_ids_2])
        else:
            updated_probs = current_model_probs
    else:
        if filter_ids_1[0].size(0) > 0:
            updated_probs = update_model_probs(current_model_probs, logits[filter_ids_1].softmax(1))
        else:
            updated_probs = current_model_probs

    # Reweight entropy losses (Eqn. 3)
    if entropys.numel() == 0:
        return outputs, 0, filter_ids_1[0].size(0), updated_probs

    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    # Diversity Regularization (IM Loss) - Matches Tent fix
    # Calculate marginal entropy on the full batch for stability
    probs_full = logits.softmax(1)
    marginal = probs_full.mean(0)
    marginal_entropy = -(marginal * torch.log(marginal + 1e-6)).sum()

    # Maximize diversity (minimize negative marginal entropy)
    loss -= 1.0 * marginal_entropy

    # Fisher regularization (Eqn. 9)
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss

    # Only step if we have valid samples and loss
    if not torch.isnan(loss):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs


def compute_fishers(model, fisher_loader, device, num_samples=None):
    """Compute Fisher Information Matrix on a dataset."""
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().to(device)

    total_samples = 0
    num_iters = 0

    for iter_, (images, _) in enumerate(fisher_loader, start=1):
        images = images.to(device)
        batch_size = images.size(0)

        if num_samples is not None and total_samples >= num_samples:
            break

        total_samples += batch_size
        num_iters += 1

        # Handle ProtoPNet output format
        outputs = model(images)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        _, targets = logits.max(1)  # Use predicted labels

        loss = train_loss_fn(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2

                fishers.update({name: [fisher, param.data.clone().detach()]})

        model.zero_grad()

    # Normalize by number of iterations
    for name in fishers:
        fishers[name][0] = fishers[name][0] / num_iters

    return fishers
