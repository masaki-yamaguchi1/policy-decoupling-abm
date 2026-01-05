# %%
# Appendix B: Figure B1 (phase-style) — aligned with Fig 2 / Tables 1–4
import numpy as np
import matplotlib.pyplot as plt
from abm import run_abm_named, BASELINE_PARAMS

# --- match baseline horizon and scenario A in Figure 2 ---
T = 120
t = np.arange(T)

T_core = 15
T_lag  = 55

def logistic_interoperability(t, I_min=0.05, I_max=1.00, kappa=0.25, t50=55):
    t = np.asarray(t, dtype=float)
    return I_min + (I_max - I_min) / (1.0 + np.exp(-kappa * (t - t50)))

# Asynchronous implementation (A): delayed convergence
I_t = logistic_interoperability(t, I_min=0.05, I_max=1.00, kappa=0.25, t50=55)
G_t = 1.0 - I_t

# --- run ABM (named output) ---
out = run_abm_named(I_t, BASELINE_PARAMS, return_states=True)

types = BASELINE_PARAMS["types"]
idx_R = types.index("R")

# Align timing: G_t is length T and corresponds to the period-t environment.
# pi_hist has length T+1; use pi_t = pi_hist[t] for t=0..T-1.
pi_t  = out["pi_hist"][:-1, :]                 # (T, 3)
pi_R  = pi_t[:, idx_R]                         # (T,)
F_bar = np.asarray(out["F_bar"], dtype=float)  # (T,)

# --- markers: start (T_core), midpoint (T_lag), end (T-1) ---
i0, im, i1 = T_core, T_lag, T - 1

# --- plot (B/W) ---
plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
})

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# (a) G_t vs pi_R,t
axes[0].invert_xaxis()
axes[0].plot(G_t, pi_R, color="black", linewidth=1.2)
axes[0].scatter(G_t, pi_R, s=8, marker="o", facecolors="none",
                edgecolors="black", linewidths=0.3)  # show time points
axes[0].scatter(G_t[i0], pi_R[i0], s=60, marker="o", facecolors="none", edgecolors="black")
axes[0].scatter(G_t[im], pi_R[im], s=60, marker="x", color="black")
axes[0].scatter(G_t[i1], pi_R[i1], s=60, marker="o", color="black")
axes[0].set_xlabel(r"Sunrise gap $G_t$")
axes[0].set_ylabel(r"Share of high-harm actors $\pi_{R,t}$")
axes[0].set_title("(a) Selection and composition")

# (b) G_t vs Fbar_t
axes[1].invert_xaxis()
axes[1].plot(G_t, F_bar, color="black", linewidth=1.2)
axes[1].scatter(G_t, F_bar, s=8, marker="o", facecolors="none",
                edgecolors="black", linewidths=0.3)
axes[1].scatter(G_t[i0], F_bar[i0], s=60, marker="o", facecolors="none", edgecolors="black")
axes[1].scatter(G_t[im], F_bar[im], s=60, marker="x", color="black")
axes[1].scatter(G_t[i1], F_bar[i1], s=60, marker="o", color="black")
axes[1].set_xlabel(r"Sunrise gap $G_t$")
axes[1].set_ylabel(r"Average harm severity $\bar F_t$")
axes[1].set_title("(b) Selection and harm")

plt.tight_layout()
plt.savefig("FigB1.png", dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.savefig("FigB1.tiff", dpi=600, bbox_inches="tight", facecolor="white", transparent=False)
plt.savefig("FigB1.eps", bbox_inches="tight", facecolor="white", transparent=False)
plt.show()



