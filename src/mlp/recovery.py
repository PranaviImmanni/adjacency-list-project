"""Recovery training after pruning: masked fine-tuning and LoRA adapters.

Both use the same budget (the follow-up config): Adam, mini-batches of
`recovery_batch_size`, `recovery_epochs` epochs, once for each learning rate
in `recovery_learning_rates`. The batch order comes from a generator seeded
with the model seed, so every learning rate sees the same batches.

Selection uses validation data only: the (learning rate, epoch) with the best
validation accuracy; ties go to the lower validation loss, then to the
earlier learning rate and epoch. Epoch 0 (before any training) is recorded
but cannot be selected, so the selected model has always been trained. The
test set is never used here.

Masked fine-tuning
    Every weight and bias is trained, but removed weights stay exactly zero:
    their gradients are zeroed before each step (so Adam's moments for them
    stay zero and they never move) and the masks are re-applied after each
    step. The retained positions never change, so the stored bytes do not
    change either.

LoRA
    The pruned weights and all biases are frozen. Each adapted layer gets
    W x + (alpha / rank) B A x, with A (rank x in) initialized like
    nn.Linear (Kaiming uniform, seeded with the model seed) and B (out x rank)
    = 0, so training starts exactly from the pruned model. Only A and B are
    trained.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gcn.config import set_seed
from src.gcn.error_analysis import exact_accuracy

from .weight_pruning import apply_mask


def _evaluate(module, x, y, batch_size):
    module.eval()
    with torch.no_grad():
        logits = torch.cat([module(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    return exact_accuracy(logits, y, torch.ones_like(y, dtype=torch.bool)), F.cross_entropy(logits, y).item()


def _better(candidate, best):
    if best is None:
        return True
    return candidate["val_accuracy"] > best["val_accuracy"] or (
        candidate["val_accuracy"] == best["val_accuracy"] and candidate["val_loss"] < best["val_loss"])


def _train(module, params, data, lr, settings, seed, device, before_step=None, after_step=None):
    """One learning rate: returns (best trained epoch with its state, history including epoch 0)."""
    x_train, y_train = data.x_train.to(device), data.y_train.to(device)
    x_val, y_val = data.x_val.to(device), data.y_val.to(device)
    batch_size, eval_batch = settings["recovery_batch_size"], settings["eval_batch_size"]
    optimizer = torch.optim.Adam(params, lr=lr)
    shuffle = torch.Generator().manual_seed(seed)

    val_acc, val_loss = _evaluate(module, x_val, y_val, eval_batch)
    history = [{"epoch": 0, "train_loss": None, "val_loss": val_loss, "val_accuracy": val_acc}]
    best = None
    for epoch in range(1, settings["recovery_epochs"] + 1):
        module.train()
        order = torch.randperm(len(y_train), generator=shuffle).to(device)
        total, count = 0.0, 0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(module(x_train[idx]), y_train[idx])
            loss.backward()
            if before_step is not None:
                before_step()
            optimizer.step()
            if after_step is not None:
                after_step()
            total += loss.item() * len(idx)
            count += len(idx)
        val_acc, val_loss = _evaluate(module, x_val, y_val, eval_batch)
        entry = {"epoch": epoch, "train_loss": total / count, "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(entry)
        if _better(entry, best):
            best = {**entry, "state": copy.deepcopy(module.state_dict())}
    return best, history


def _select(runs):
    """runs: list of (lr, best, history) in config order -> (lr, best)."""
    chosen = None
    for lr, best, _ in runs:
        if chosen is None or _better(best, chosen[1]):
            chosen = (lr, best)
    return chosen


def _record(kind, runs, chosen, settings, extra=None):
    lr, best = chosen
    return {
        "method": kind,
        "selection": "best validation accuracy over (learning rate, epoch >= 1); ties -> lower validation loss, "
                     "then earlier learning rate and epoch",
        "selected": {"learning_rate": lr, "epoch": best["epoch"], "val_accuracy": best["val_accuracy"],
                     "val_loss": best["val_loss"]},
        "budget": {"epochs": settings["recovery_epochs"], "learning_rates": settings["recovery_learning_rates"],
                   "batch_size": settings["recovery_batch_size"], "optimizer": "Adam", "weight_decay": 0.0},
        "runs": [{"learning_rate": lr_, "best_epoch": b["epoch"], "best_val_accuracy": b["val_accuracy"],
                  "best_val_loss": b["val_loss"], "history": h} for lr_, b, h in runs],
        **(extra or {}),
    }


def masked_finetune(model, masks, data, settings, seed, device="cpu"):
    """Fine-tune `model` with `masks` fixed. Returns (fine-tuned MLP on CPU, record)."""
    device = torch.device(device)
    runs = []
    for lr in settings["recovery_learning_rates"]:
        set_seed(seed)
        tuned = copy.deepcopy(model).to(device)
        masks_dev = {name: mask.to(device) for name, mask in masks.items()}
        weights = {name: getattr(tuned, name).weight for name in masks}
        with torch.no_grad():
            for name, w in weights.items():
                w.copy_(apply_mask(w, masks_dev[name]))

        def zero_removed_grads():
            for name, w in weights.items():
                w.grad.mul_(masks_dev[name])

        def reapply_masks():
            with torch.no_grad():
                for name, w in weights.items():
                    w.mul_(masks_dev[name])

        best, history = _train(tuned, tuned.parameters(), data, lr, settings, seed, device,
                               zero_removed_grads, reapply_masks)
        runs.append((lr, best, history))

    chosen = _select(runs)
    result = copy.deepcopy(model).cpu()
    result.load_state_dict({k: v.cpu() for k, v in chosen[1]["state"].items()})
    for name, mask in masks.items():
        if bool(getattr(result, name).weight.detach()[~mask].abs().sum() != 0):
            raise RuntimeError(f"{name}: a removed weight became nonzero during fine-tuning")
    return result.eval(), _record("masked fine-tuning", runs, chosen, settings,
                                  {"trained": "all weights and biases; removed weights fixed at zero"})


class LoRAMLP(nn.Module):
    """Frozen (pruned) MLP plus trainable low-rank adapters on `layers`."""

    def __init__(self, model, masks, rank, alpha, layers):
        super().__init__()
        self.names = list(model.layer_names)
        self.scale = alpha / rank
        self.lora_a, self.lora_b = nn.ParameterDict(), nn.ParameterDict()
        for name in self.names:
            linear = getattr(model, name)
            weight = linear.weight.detach().clone()
            if name in masks:
                weight = apply_mask(weight, masks[name])
            self.register_buffer(f"{name}_weight", weight)
            self.register_buffer(f"{name}_bias", linear.bias.detach().clone())
            if name in layers:
                out_features, in_features = weight.shape
                a = torch.empty(rank, in_features)
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                self.lora_a[name] = nn.Parameter(a)
                self.lora_b[name] = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        h, last = x, len(self.names) - 1
        for i, name in enumerate(self.names):
            out = F.linear(h, getattr(self, f"{name}_weight"), getattr(self, f"{name}_bias"))
            if name in self.lora_a:
                out = out + self.scale * F.linear(F.linear(h, self.lora_a[name]), self.lora_b[name])
            h = torch.relu(out) if i < last else out
        return h

    def adapters(self):
        """{layer: (A, scale * B)} on CPU, the form the storage formats take."""
        return {name: (self.lora_a[name].detach().cpu().clone(),
                       (self.scale * self.lora_b[name]).detach().cpu().clone()) for name in self.lora_a}


def lora_recover(model, masks, data, settings, seed, device="cpu"):
    """Train rank-r adapters on the frozen pruned model. Returns (adapters on CPU, record)."""
    device = torch.device(device)
    rank, alpha, layers = settings["lora_rank"], settings["lora_alpha"], settings["lora_layers"]
    runs = []
    for lr in settings["recovery_learning_rates"]:
        set_seed(seed)  # same adapter initialization for every learning rate
        module = LoRAMLP(model, masks, rank, alpha, layers).to(device)
        params = list(module.lora_a.parameters()) + list(module.lora_b.parameters())
        best, history = _train(module, params, data, lr, settings, seed, device)
        runs.append((lr, best, history))

    chosen = _select(runs)
    module = LoRAMLP(model, masks, rank, alpha, layers)
    module.load_state_dict({k: v.cpu() for k, v in chosen[1]["state"].items()})
    params = sum(p.numel() for p in module.lora_a.parameters()) + sum(p.numel() for p in module.lora_b.parameters())
    return module.adapters(), _record("LoRA", runs, chosen, settings, {
        "trained": "adapters only; pruned weights and biases frozen",
        "rank": rank, "alpha": alpha, "scale": alpha / rank, "layers": list(layers), "adapter_parameters": params})
