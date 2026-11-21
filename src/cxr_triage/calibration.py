"""Per-label temperature scaling (Guo et al., 2017), fit on the validation set."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, labels: np.ndarray, iters: int = 200) -> np.ndarray:
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.detach().exp().numpy()


def apply_temperature(logits: np.ndarray, temps: np.ndarray) -> np.ndarray:
    return logits / temps
