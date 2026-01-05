# %%
# Figure 4: Robustness and sensitivity (REPE)
# Rewritten version consistent with the paper's notion of "synchronization".
# ---------------------------------------------------------------
# Key change vs earlier (problematic) version:
#   - We DO NOT define S as I_t ≡ 1 (immediate full implementation).
#   - Instead, we hold the SAME transition path I_t for A and S,
#     and isolate "synchronization" by equalizing channel sensitivities:
#         A (async): alpha_A != alpha_B
#         S (sync):  alpha_A = alpha_B = (alpha_A + alpha_B)/2
#   - Thus ΔF̄ = mean(F̄_A - F̄_S) over the transition window captures the
#     pure effect of channel-asynchrony (implementation synchronicity),
#     matching the interpretation used in Figure C1.
#
# Outputs:
#   figure4.eps, figure4.tiff (600dpi), figure4.png (300dpi)

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
# Time & policy path (same for A and S)
# =========================
T = 100
T0, T1 = 30, 60
TRANSITION_WINDOW = range(T0, T1)  # 30..59

# Linear interoperability transition (same as baseline Figure 2 design)
I_t = np.ones(T)
I_t[:T0] = 0.0
I_t[T0:T1] = np.linspace(0.0, 1.0, T1 - T0)

# =========================
# Experiment settings
# =========================
PI_R_GRID = np.linspace(0.01, 0.30, 15)
FR_FC_GRID = np.linspace(1.0, 3.0, 15)

N_REPS_A = 30
N_REPS_B = 20
DIRICHLET_SCALE = 50

rng = np.random.default_rng(123)

# =========================
# Regimes: A vs S (synchronization = alpha equalization)
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
# Helper: ΔF̄ over transition window
# =========================

def delta_Fbar(I_t, pA, pS, window) -> float:
    out_A = run_abm_named(I_t, pA, return_states=False)
    out_S = run_abm_named(I_t, pS, return_states=False)

    Fbar_A = np.asarray(out_A["F_bar"], dtype=float)
    Fbar_S = np.asarray(out_S["F_bar"], dtype=float)

    T_eff = min(len(Fbar_A), len(Fbar_S), len(I_t))
    w = np.array([t for t in window if 0 <= t < T_eff], dtype=int)
    return float(np.mean(Fbar_A[w]) - np.mean(Fbar_S[w]))


# =========================
# Run experiments
# =========================

x_a, y_a = [], []
x_b, y_b = [], []

# Panel (a): initial composition sensitivity (Dirichlet perturbations)
for piR in PI_R_GRID:
    for _ in range(N_REPS_A):
        alpha_dir = np.array([0.7, 0.2, piR], dtype=float)
        alpha_dir /= alpha_dir.sum()
        pi0 = rng.dirichlet(DIRICHLET_SCALE * alpha_dir)

        pA = params_async(BASELINE_PARAMS)
        pS = params_sync(BASELINE_PARAMS)

        # use the same randomized initial composition in both regimes
        pA["pi0"] = {"O": float(pi0[0]), "C": float(pi0[1]), "R": float(pi0[2])}
        pS["pi0"] = {"O": float(pi0[0]), "C": float(pi0[1]), "R": float(pi0[2])}

        x_a.append(float(piR))
        y_a.append(delta_Fbar(I_t, pA, pS, TRANSITION_WINDOW))

# Panel (b): relax severity ordering via F_R/F_C
F_C = BASELINE_PARAMS["F"]["C"]
for ratio in FR_FC_GRID:
    for _ in range(N_REPS_B):
        pA = params_async(BASELINE_PARAMS)
        pS = params_sync(BASELINE_PARAMS)

        # vary only F_R (keep same in both A and S)
        pA["F"] = pA["F"].copy()
        pS["F"] = pS["F"].copy()
        pA["F"]["R"] = float(ratio * F_C)
        pS["F"]["R"] = float(ratio * F_C)

        x_b.append(float(ratio))
        y_b.append(delta_Fbar(I_t, pA, pS, TRANSITION_WINDOW))

x_a, y_a = np.array(x_a), np.array(y_a)
x_b, y_b = np.array(x_b), np.array(y_b)

# =========================
# Plot (black & white)
# =========================
plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(x_a, y_a, s=15, color="black", alpha=0.6)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xlabel(r"Initial $\pi_{R,0}$")
axes[0].set_ylabel(r"$\Delta \bar F$ (A$-$S)")
axes[0].set_title("(a) Initial composition sensitivity")

axes[1].scatter(x_b, y_b, s=15, color="black", alpha=0.6)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xlabel(r"$F_R/F_C$")
axes[1].set_ylabel(r"$\Delta \bar F$ (A$-$S)")
axes[1].set_title("(b) Relaxing severity ordering")

plt.tight_layout()

# =========================
# Save in multiple formats
# =========================
plt.savefig("fig4.eps", format="eps")
plt.savefig("fig4.tiff", format="tiff", dpi=600)
plt.savefig("fig4.png", format="png", dpi=300)
plt.show()

# =========================
# Numeric summary (rounded)
# =========================

def r2(x):
    return np.round(x, 2)

print("Panel (a): share ΔF̄>0 =", r2(np.mean(y_a > 0)))
print("Panel (b): share ΔF̄>0 =", r2(np.mean(y_b > 0)))

print("Panel (a): ΔF̄ mean (min, max) =",
      r2(np.mean(y_a)), (r2(np.min(y_a)), r2(np.max(y_a))))

print("Panel (b): ΔF̄ mean (min, max) =",
      r2(np.mean(y_b)), (r2(np.min(y_b)), r2(np.max(y_b))))



