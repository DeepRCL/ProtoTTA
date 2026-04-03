from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tent import collect_params, configure_model, copy_model_and_optimizer, load_model_and_optimizer, softmax_entropy

class EATA(nn.Module):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, fishers=None, fisher_alpha=2000.0, steps=1, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.fishers = fishers # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = fisher_alpha # trade-off \beta for two losses (Eqn. 8) 

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        # Adaptation tracking statistics
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,  # EATA only adapts filtered samples
            'total_updates': 0,    # Total optimizer steps
        }

    def forward(self, x):
        if self.episodic:
            self.reset()
        
        # Track adaptation
        batch_size = x.size(0)
        self.adaptation_stats['total_samples'] += batch_size
        
        outputs, min_distances, values = None, None, None
        
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, min_distances, values, num_counts_2, num_counts_1, updated_probs = \
                    forward_and_adapt_eata(x, self.model, self.optimizer, self.fishers, 
                                          self.e_margin, self.current_model_probs, 
                                          fisher_alpha=self.fisher_alpha, 
                                          d_margin=self.d_margin)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
                
                # Track actual adaptation
                self.adaptation_stats['adapted_samples'] += num_counts_2  # Only samples that passed filtering
                if num_counts_2 > 0:
                    # One update per adapted sample per step
                    self.adaptation_stats['total_updates'] += num_counts_2
        else:
            self.model.eval()
            with torch.no_grad():
                outputs, min_distances, values = self.model(x)
                
        return outputs, min_distances, values

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        # Only reset model parameters, NOT optimizer state
        # Preserves Adam momentum/variance for effective episodic updates
        self.model.load_state_dict(self.model_state, strict=True)
        self.current_model_probs = None

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    def forward_no_adapt(self, x):
        """Forward pass without adaptation (used for metrics)."""
        return self.model(x)

    def __getattr__(self, name):
        """Forward attribute access to the underlying model if not found in EATA."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def update_model_probs(current_model_probs, new_probs):
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
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=2000.0, d_margin=0.05):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs, min_distances, values = model(x)
    
    # adapt
    entropys = softmax_entropy(outputs)
    
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0] > -0.1)
    entropys = entropys[filter_ids_1]
    
    # filter redundant samples
    if current_model_probs is not None:
        # Use softmax of outputs for cosine similarity
        probs = outputs.softmax(1)
        # Check if we have any samples left
        if filter_ids_1[0].size(0) > 0:
            cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), probs[filter_ids_1], dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(current_model_probs, probs[filter_ids_1][filter_ids_2])
        else:
             updated_probs = current_model_probs
    else:
        if filter_ids_1[0].size(0) > 0:
            updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
        else:
            updated_probs = current_model_probs

    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
        
    # Only step if we have loss and samples (and loss is not NaN)
    # If num samples is 0, loss will be nan (mean of empty).
    if entropys.numel() > 0 and not torch.isnan(loss):
        loss.backward()
        optimizer.step()
    
    optimizer.zero_grad()
    
    return outputs, min_distances, values, entropys.size(0), filter_ids_1[0].size(0), updated_probs


def compute_fishers(model, fisher_loader, device, num_samples=None):
    """Compute Fisher Information Matrix on a dataset (clean or test)."""
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Ensure model is in correct mode (usually train mode for BN/LN adaptation)
    
    total_samples = 0
    num_iters = 0
    
    for iter_, (images, _) in enumerate(fisher_loader, start=1): # Ignore targets, use preds
        images = images.to(device)
        batch_size = images.size(0)
        
        if num_samples is not None and total_samples >= num_samples:
            break
            
        total_samples += batch_size
        num_iters += 1
        
        outputs, _, _ = model(images) 
        _, targets = outputs.max(1) # Use predicted labels
        
        loss = train_loss_fn(outputs, targets)
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

