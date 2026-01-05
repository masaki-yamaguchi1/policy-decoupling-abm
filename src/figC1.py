# %%
# Figure C1: Robustness to alternative implementation paths
# Clean version using abm.run_abm_named (no tuple indexing)
# -----------------------------------------------------------
# Interpretation:
#   A (asynchronous): channel sensitivities differ (alpha_A != alpha_B)
#   S (synchronized): channel sensitivities are equalized (alpha_A = alpha_B)
#   I_t path varies only in the transition segment; window matches Fig.4.
#
# Outputs:
#   figureC1.eps, figureC1.tiff (600dpi), figureC1.png (300dpi)

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from abm import run_abm_named

# =========================
# Baseline parameters
# =========================
BASELINE_PARAMS = {
    "types": ["O", "C", "R"],
    "channels": ["A", "B", "C"],
    "lambda": {"O": 0.5, "C": 1.0, "R": 2.0},
    "mu":     {"O": 2.0, "C": 1.0, "R": 0.5},
    "F":      {"O": 1.0, "C": 3.0, "R": 6.0},
    "pbar":   {"A": 0.10, "B": 0.06, "C": 0.02},
    "alpha":  {"A": 0.4, "B": 0.8, "C": 0.0},
    "c":      {"A": 0.5, "B": 0.8, "C": 1.5},
    "pi0":    {"O": 0.70, "C": 0.20, "R": 0.10},
    "N0":     1000,
}

# =========================
# Time & transition window (MATCH Figure 4)
# =========================
T = 100
T0, T1 = 30, 60
TRANSITION_WINDOW = range(T0, T1)  # 30..59

# =========================
# Alternative I_t paths (ONLY [T0,T1) differs)
# =========================

def build_transition_segment(kind: str, n: int) -> np.ndarray:
    """Construct I_t segment on [T0,T1): 0->1 with different shapes."""
    if n <= 1:
        return np.array([1.0], dtype=float)

    if kind == "Linear":
        return np.linspace(0.0, 1.0, n)

    if kind == "Front-loaded":
        n1 = n // 2
        n2 = n - n1
        mid = 0.70
        seg1 = np.linspace(0.0, mid, n1, endpoint=False)
        seg2 = np.linspace(mid, 1.0, n2, endpoint=True)
        return np.concatenate([seg1, seg2])

    if kind == "Back-loaded":
        n1 = n // 2
        n2 = n - n1
        mid = 0.30
        seg1 = np.linspace(0.0, mid, n1, endpoint=False)
        seg2 = np.linspace(mid, 1.0, n2, endpoint=True)
        return np.concatenate([seg1, seg2])

    if kind == "Stepwise":
        n1 = n // 3
        n2 = n // 3
        n3 = n - n1 - n2
        seg = np.concatenate([
            np.full(n1, 0.0),
            np.full(n2, 0.5),
            np.full(n3, 1.0),
        ]).astype(float)
        seg[0] = 0.0
        seg[-1] = 1.0
        return seg

    if kind == "Plateau":
        n1 = n // 3
        n2 = n // 3
        n3 = n - n1 - n2
        mid = 0.55
        seg1 = np.linspace(0.0, mid, n1, endpoint=False) if n1 > 0 else np.array([], float)
        seg2 = np.full(n2, mid, dtype=float)
        seg3 = np.linspace(mid, 1.0, n3, endpoint=True) if n3 > 0 else np.array([], float)
        seg = np.concatenate([seg1, seg2, seg3])
        seg[0] = 0.0
        seg[-1] = 1.0
        return seg

    raise ValueError(f"Unknown kind: {kind}")


def make_I_path(kind: str, T: int, T0: int, T1: int) -> np.ndarray:
    """Full I_t: 0 pre, shaped transition, 1 post."""
    I = np.ones(T, dtype=float)
    I[:T0] = 0.0
    I[T0:T1] = build_transition_segment(kind, T1 - T0)
    I[T1:] = 1.0
    return np.clip(I, 0.0, 1.0)


# =========================
# Regimes: A vs S (alpha synchronization benchmark)
# =========================

def params_async(base: dict) -> dict:
    return deepcopy(base)


def params_sync(base: dict) -> dict:
    p = deepcopy(base)
    aA, aB = p["alpha"]["A"], p["alpha"]["B"]
    a = 0.5 * (aA + aB)
    p["alpha"] = p["alpha"].copy()
    p["alpha"]["A"] = a
    p["alpha"]["B"] = a
    return p


# =========================
# Statistic: transition-period Delta Fbar
# =========================

def delta_Fbar(I_t: np.ndarray, pA: dict, pS: dict, window) -> float:
    outA = run_abm_named(I_t, pA, return_states=False)
    outS = run_abm_named(I_t, pS, return_states=False)
    FbarA = np.asarray(outA["F_bar"], dtype=float)
    FbarS = np.asarray(outS["F_bar"], dtype=float)

    T_eff = min(len(FbarA), len(FbarS), len(I_t))
    w = np.array([t for t in window if 0 <= t < T_eff], dtype=int)
    return float(np.mean(FbarA[w]) - np.mean(FbarS[w]))


# =========================
# Run and plot
# =========================
path_kinds = ["Linear", "Front-loaded", "Back-loaded", "Stepwise", "Plateau"]
vals = []

pA = params_async(BASELINE_PARAMS)
pS = params_sync(BASELINE_PARAMS)

for kind in path_kinds:
    I_t = make_I_path(kind, T, T0, T1)
    vals.append(delta_Fbar(I_t, pA, pS, TRANSITION_WINDOW))

vals = np.array(vals, dtype=float)

print("Delta Fbar (A-S) by path (alpha-synchronization benchmark):")
for k, v in zip(path_kinds, vals):
    print(f"  {k:12s}: {v:.4f}")

# Plot (black & white)
plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.2))

x = np.arange(len(path_kinds))
ax.bar(x, vals, color="white", edgecolor="black", linewidth=1.2)
ax.axhline(0.0, color="black", linewidth=1.0)

ax.set_xticks(x)
ax.set_xticklabels(path_kinds, rotation=15, ha="right")
ax.set_ylabel(r"Transition-period $\Delta \bar F$ (A$-$S)")
ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.2f}")

plt.tight_layout()

plt.savefig("figC1.eps", format="eps")
plt.savefig("figC1.tiff", format="tiff", dpi=600)
plt.savefig("figC1.png", format="png", dpi=300)
plt.show()



