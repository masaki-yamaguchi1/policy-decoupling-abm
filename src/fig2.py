# %%
# fig2_main_fixed.py
import numpy as np
import matplotlib.pyplot as plt
from abm import run_abm_named, BASELINE_PARAMS

# ============================================================
# 1) Interoperability paths
# ============================================================
def logistic_interoperability(t, I_min=0.05, I_max=1.0, kappa=0.25, t50=55.0):
    t = np.asarray(t, dtype=float)
    return I_min + (I_max - I_min) / (1.0 + np.exp(-kappa * (t - t50)))

T = 120
t = np.arange(T)

T_core = 15
T_lag  = 55

# A / S share the SAME I_t (per your confirmed benchmark logic)
I_base  = logistic_interoperability(t, I_min=0.05, I_max=1.00, kappa=0.25, t50=55)
I_paper = logistic_interoperability(t, I_min=0.05, I_max=0.70, kappa=0.25, t50=55)

# ============================================================
# 2) Parameters (CHOOSE ONE SOURCE OF TRUTH)
# ============================================================
USE_TABLE_PARAMS = False  # <- 最終稿ではどちらかに固定

if USE_TABLE_PARAMS:
    params_A = {
        "types": ["O","C","R"],
        "channels": ["A","B","C"],
        "lambda": {"O": 0.6, "C": 1.0, "R": 1.4},
        "mu":     {"O": 1.2, "C": 0.8, "R": 0.6},
        "F":      {"O": 1.0, "C": 3.0, "R": 6.0},
        "pbar":   {"A": 0.05, "B": 0.03, "C": 0.01},
        "alpha":  {"A": 0.4, "B": 0.7,  "C": 0.0},
        "c":      {"A": 1.0, "B": 0.6,  "C": 0.3},
        "pi0":    {"O": 0.6, "C": 0.3,  "R": 0.1},
        "N0": 1000,
    }
else:
    # NOTE: この場合は Table 1–3 を BASELINE_PARAMS に合わせて更新が必要
    params_A = BASELINE_PARAMS

# S: alpha synchronization benchmark (alpha_B = alpha_A), with the SAME I_t
params_S = dict(params_A)
params_S["alpha"] = dict(params_A["alpha"])
params_S["alpha"]["B"] = params_S["alpha"]["A"]

# ============================================================
# 3) Run ABM (named outputs)
# ============================================================
out_S  = run_abm_named(I_base,  params_S)
out_A  = run_abm_named(I_base,  params_A)
out_Ap = run_abm_named(I_paper, params_A)

# sanity checks
for out in (out_S, out_A, out_Ap):
    assert len(out["Y"]) == T and len(out["F_bar"]) == T and len(out["H"]) == T
    assert np.allclose(out["H"], out["Y"] * out["F_bar"])

# ============================================================
# 4) Plot (REPE-style, B&W, no titles/captions)
# ============================================================
width_mm, height_mm = 174, 65
figsize_in = (width_mm / 25.4, height_mm / 25.4)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
})

fig, axes = plt.subplots(1, 3, figsize=figsize_in, sharex=True)

style_S  = dict(color="black", linestyle="-")
style_A  = dict(color="black", linestyle="--")
style_Ap = dict(color="black", linestyle=":")

def add_policy_timing(ax):
    ax.axvline(T_core, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(T_lag,  color="black", linestyle="--", linewidth=0.8)

# (a) Quantity
ax = axes[0]
ax.plot(t, out_S["Y"],  label="S",  **style_S)
ax.plot(t, out_A["Y"],  label="A",  **style_A)
ax.plot(t, out_Ap["Y"], label="A'", **style_Ap)
ax.set_ylabel(r"Illicit volume $Y_t$")
add_policy_timing(ax)
ax.legend(frameon=False, loc="center right")
ax.text(-0.15, 1.05, "(a)", transform=ax.transAxes, ha="left", va="bottom")

# (b) Quality
ax = axes[1]
ax.plot(t, out_S["F_bar"],  **style_S)
ax.plot(t, out_A["F_bar"],  **style_A)
ax.plot(t, out_Ap["F_bar"], **style_Ap)
ax.set_ylabel(r"Avg. harm severity $\bar{F}_t$")
add_policy_timing(ax)
ax.text(-0.15, 1.05, "(b)", transform=ax.transAxes, ha="left", va="bottom")

# (c) Social loss
ax = axes[2]
ax.plot(t, out_S["H"],  **style_S)
ax.plot(t, out_A["H"],  **style_A)
ax.plot(t, out_Ap["H"], **style_Ap)
ax.set_ylabel(r"Social loss $H_t = Y_t \cdot \bar{F}_t$")
ax.set_xlabel(r"Time $t$")
add_policy_timing(ax)
ax.text(-0.15, 1.05, "(c)", transform=ax.transAxes, ha="left", va="bottom")

for ax in axes:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)

fig.tight_layout(rect=[0.08, 0, 0.90, 1])

# White background (no transparency)
fig.savefig("Fig2.eps",  format="eps",  facecolor="white", transparent=False)
fig.savefig("Fig2.tiff", format="tiff", dpi=1200, facecolor="white", transparent=False)
fig.savefig("Fig2.png",  dpi=300, facecolor="white", transparent=False)

plt.show()



