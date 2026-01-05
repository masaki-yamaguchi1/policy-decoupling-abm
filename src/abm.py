# -*- coding: utf-8 -*-
"""abm.py

Deterministic, expectation-based (mean-field) ABM used in the REPE paper.

This rewrite keeps full backward compatibility with your existing notebooks,
while eliminating the 'out[i]' ambiguity by providing a named-return wrapper.

Core model (per period t):
  - Sunrise gap:          G_t = 1 - I_t
  - Detection hazards:    p_{k,t} = pbar_k (1 - alpha_k G_t)   (alpha_C may be 0)
  - Expected cost:        EC_{θ,k,t} = λ_θ p_{k,t} F_θ + μ_θ c_k
  - Logit choice:         P_{θ,k,t} ∝ exp(-EC_{θ,k,t})
  - Exit (detection):     d_{θ,t} = Σ_k P_{θ,k,t} p_{k,t}
  - Replicator update:    π_{θ,t+1} ∝ (1-d_{θ,t}) π_{θ,t}
  - Total actors:         N_{t+1} = N_t (1-d̄_t),  d̄_t = Σ_θ π_{θ,t} d_{θ,t}
  - Successful volume:    Y_t = N_t (1-d̄_t)
  - Avg harm severity:    F̄_t = Σ_θ π_{θ,t} F_θ
  - Social loss:          H_t = Y_t F̄_t

Recommended usage for figures:
  - Use run_abm_named(...) to get a dict with keys: "Y", "F_bar", "H", "pi_hist", "N_hist".
  - This avoids any reliance on tuple index ordering.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike1D = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class ABMResult:
    """Named container for ABM outputs."""

    Y: np.ndarray                 # (T,)
    F_bar: np.ndarray             # (T,)
    H: np.ndarray                 # (T,)
    pi_hist: Optional[np.ndarray] # (T+1, n_types) or None
    N_hist: Optional[np.ndarray]  # (T+1,) or None

    def as_dict(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            "Y": self.Y,
            "F_bar": self.F_bar,
            "H": self.H,
            "pi_hist": self.pi_hist,
            "N_hist": self.N_hist,
        }


def run_abm(
    I_t: ArrayLike1D,
    params: Dict,
    *,
    return_states: bool = False,
    return_named: bool = False,
    eps: float = 1e-12,
):
    """Run the ABM.

    Backward compatible behavior:
      - return_states=False, return_named=False  -> returns (Y, Fbar, H)
      - return_states=True,  return_named=False  -> returns (Y, Fbar, H, pi_hist, N_hist)

    New behavior (recommended):
      - return_named=True -> returns ABMResult (named outputs)

    Parameters
    ----------
    I_t : array-like, shape (T,)
        Interoperability path in [0,1].
    params : dict
        Must include keys:
          - "types", "channels", "lambda", "mu", "F", "pbar", "alpha", "c", "pi0", "N0"
    return_states : bool
        If True, compute and return (pi_hist, N_hist).
    return_named : bool
        If True, return ABMResult instead of tuples.
    eps : float
        Numerical floor.
    """

    I_t = np.asarray(I_t, dtype=float)
    T = int(I_t.size)

    # --- Unpack and validate ---
    types: List[str] = list(params["types"])
    channels: List[str] = list(params["channels"])

    n_types = len(types)
    n_channels = len(channels)

    lam = params["lambda"]
    mu = params["mu"]
    F = params["F"]
    pbar = params["pbar"]
    alpha = params["alpha"]
    c = params["c"]
    pi0 = params["pi0"]
    N0 = float(params["N0"])

    lam_vec = np.array([float(lam[th]) for th in types], dtype=float)     # (n_types,)
    mu_vec = np.array([float(mu[th]) for th in types], dtype=float)       # (n_types,)
    F_vec = np.array([float(F[th]) for th in types], dtype=float)         # (n_types,)

    c_vec = np.array([float(c[k]) for k in channels], dtype=float)        # (n_channels,)
    pbar_vec = np.array([float(pbar[k]) for k in channels], dtype=float)  # (n_channels,)
    alpha_vec = np.array([float(alpha.get(k, 0.0)) for k in channels], dtype=float)  # (n_channels,)

    pi = np.array([float(pi0[th]) for th in types], dtype=float)

    if not np.isclose(pi.sum(), 1.0):
        raise ValueError(f"pi0 must sum to 1.0, got {pi.sum():.6f}")
    if np.any(pi < 0):
        raise ValueError("pi0 must be nonnegative.")
    if not (0.0 <= float(I_t.min()) and float(I_t.max()) <= 1.0):
        raise ValueError("I_t must lie in [0,1].")

    # Histories
    pi_hist = None
    N_hist = None
    if return_states:
        pi_hist = np.zeros((T + 1, n_types), dtype=float)
        N_hist = np.zeros(T + 1, dtype=float)
        pi_hist[0, :] = pi
        N_hist[0] = N0

    # Outputs
    Y = np.zeros(T, dtype=float)
    Fbar = np.zeros(T, dtype=float)
    H = np.zeros(T, dtype=float)

    N = N0

    for t in range(T):
        G = 1.0 - I_t[t]

        # Channel detection hazards
        p_k = pbar_vec * (1.0 - alpha_vec * G)
        p_k = np.clip(p_k, eps, 1.0)

        # Expected costs EC_{θ,k,t}
        term_risk = (lam_vec[:, None] * F_vec[:, None]) * p_k[None, :]
        term_fric = mu_vec[:, None] * c_vec[None, :]
        EC = term_risk + term_fric  # (n_types, n_channels)

        # Logit probabilities (row-stabilized)
        row_min = EC.min(axis=1, keepdims=True)
        logits = np.exp(-(EC - row_min))
        denom = logits.sum(axis=1, keepdims=True)
        P = logits / np.clip(denom, eps, None)

        # Type exit rates
        d_theta = (P * p_k[None, :]).sum(axis=1)
        d_theta = np.clip(d_theta, 0.0, 1.0)

        # Avg exit rate
        dbar = float((pi * d_theta).sum())
        dbar = min(max(dbar, 0.0), 1.0)

        # Outputs
        Y[t] = N * (1.0 - dbar)
        Fbar[t] = float((pi * F_vec).sum())
        H[t] = Y[t] * Fbar[t]

        # Replicator update
        survivors = (1.0 - d_theta) * pi
        ssum = float(survivors.sum())
        if ssum <= eps:
            pi_next = pi.copy()
        else:
            pi_next = survivors / ssum

        N_next = N * (1.0 - dbar)

        # Commit
        pi = np.clip(pi_next, 0.0, None)
        pi = pi / np.clip(pi.sum(), eps, None)
        N = max(float(N_next), 0.0)

        if return_states:
            assert pi_hist is not None and N_hist is not None
            pi_hist[t + 1, :] = pi
            N_hist[t + 1] = N

    if return_named:
        return ABMResult(Y=Y, F_bar=Fbar, H=H, pi_hist=pi_hist, N_hist=N_hist)

    # Backward compatible returns
    if return_states:
        return Y, Fbar, H, pi_hist, N_hist
    return Y, Fbar, H


def run_abm_named(
    I_t: ArrayLike1D,
    params: Dict,
    *,
    return_states: bool = False,
    eps: float = 1e-12,
) -> Dict[str, Optional[np.ndarray]]:
    """Convenience wrapper returning a dict with stable, named keys.

    Keys:
      - "Y": (T,)
      - "F_bar": (T,)
      - "H": (T,)
      - "pi_hist": (T+1, n_types) if return_states else None
      - "N_hist": (T+1,) if return_states else None

    This is the recommended interface for figure scripts.
    """
    res: ABMResult = run_abm(I_t, params, return_states=return_states, return_named=True, eps=eps)
    return res.as_dict()


# -----------------------------
# Example parameter dictionary (baseline, REPE-ready)
# -----------------------------
BASELINE_PARAMS = {
    "types": ["O", "C", "R"],
    "channels": ["A", "B", "C"],
    # O is risk-insensitive but friction-sensitive; R is risk-sensitive but friction-tolerant.
    "lambda": {"O": 0.5, "C": 1.0, "R": 2.0},
    "mu":     {"O": 2.0, "C": 1.0, "R": 0.5},

    # Private penalty severity (ordinal restriction in the paper)
    "F": {"O": 1.0, "C": 3.0, "R": 6.0},

    # Baseline detection hazards
    "pbar": {"A": 0.10, "B": 0.06, "C": 0.02},

    # Sunrise-gap sensitivity (asynchrony across channels)
    "alpha": {"A": 0.4, "B": 0.8, "C": 0.0},

    # Channel frictions
    "c": {"A": 0.5, "B": 0.8, "C": 1.5},

    # Initial composition
    "pi0": {"O": 0.70, "C": 0.20, "R": 0.10},
    "N0": 1000,
}


__all__ = [
    "ABMResult",
    "run_abm",
    "run_abm_named",
    "BASELINE_PARAMS",
]
