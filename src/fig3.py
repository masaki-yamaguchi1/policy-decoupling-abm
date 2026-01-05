# %%
# Fig3_selection_singlepanel_T300_Aonly.py
import numpy as np
import matplotlib.pyplot as plt

from abm import run_abm_named, BASELINE_PARAMS

# -----------------------------
# Horizon & timing
# -----------------------------
T = 300
T_core = 15
T_lag  = 55

def logistic_interoperability(t, I_min=0.05, I_max=1.0, kappa=0.25, t50=55.0):
    t = np.asarray(t, dtype=float)
    return I_min + (I_max - I_min) / (1.0 + np.exp(-kappa * (t - t50)))

t_period = np.arange(T)                 # periods 0..T-1
I_A = logistic_interoperability(t_period, I_min=0.05, I_max=1.0, kappa=0.25, t50=55.0)

# -----------------------------
# Run ABM (A only)
# -----------------------------
out_A = run_abm_named(I_A, BASELINE_PARAMS, return_states=True)
pi_hist = out_A["pi_hist"]              # (T+1, 3) including t=0 initial

t_state = np.arange(T + 1)              # states at t=0..T
pi_O, pi_C, pi_R = pi_hist[:, 0], pi_hist[:, 1], pi_hist[:, 2]

# -----------------------------
# Plot (single panel, B&W)
# -----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.plot(t_state, pi_O, color="black", linestyle="-",  linewidth=2.6, label="O")
ax.plot(t_state, pi_C, color="black", linestyle="--", linewidth=2.6, label="C")
ax.plot(t_state, pi_R, color="black", linestyle=":",  linewidth=3.2, label="R")

# adoption timing markers
ax.axvline(T_core, color="black", linestyle="--", linewidth=1.0)
ax.axvline(T_lag,  color="black", linestyle="--", linewidth=1.0)

ax.set_xlim(0, T)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Time $t$")
ax.set_ylabel("Population share $\\pi_{\\theta,t}$")

# cosmetics
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="out", length=3, width=0.8)

ax.legend(frameon=False, loc="center right")

# panel label (optional)
ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, ha="left", va="top")

fig.tight_layout()

# Export (white background, non-transparent)
fig.savefig("Fig3.png",  dpi=300, facecolor="white", transparent=False)
fig.savefig("Fig3.tiff", dpi=1200, format="tiff", facecolor="white", transparent=False)
fig.savefig("Fig3.eps",  format="eps", facecolor="white", transparent=False)

plt.show()



