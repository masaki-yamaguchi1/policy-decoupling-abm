# Policy Decoupling in Asynchronous Travel Rule Implementation (ABM)

This repository contains the simulation code used in the paper on asynchronous
Travel Rule implementation, the sunrise gap, and the Policy Decoupling Effect.

The code implements an agent-based model (ABM) with heterogeneous illicit actors
choosing across transaction channels under evolving interoperability. During the
transition, selection (exit via detection / frictions) changes the composition of
active actor types and can intensify average harm severity even when aggregate
activity contracts.

## Repository policy (important)

- The repository tracks **source code and documentation only**.
- Generated outputs (figures/tables) should be written under `outputs/` and are
  **not** committed to version control.

## Files

All scripts are under `src/`.

- `src/abm.py`  
  Core ABM implementation. Figure scripts call `run_abm_named(I_t, params)`,
  which returns:
  `{ "Y", "F_bar", "H", "pi_hist", "N_hist" }`.

- `src/fig2.py`  
  Reproduces **Figure 2**: scenarios **S / A / A′**  
  - S: early convergence  
  - A: delayed convergence  
  - A′: lower `I_max`

- `src/fig3.py`  
  Reproduces **Figure 3** (scenario **A** only, single panel).  
  **Note:** Table 3 uses `T=120`, while Figure 3 uses `T=300` for visualization.
  The caption should state this explicitly. Do not claim increases in absolute
  levels; interpret changes in **composition/shares** only.

- `src/fig4.py`  
  Reproduces **Figure 4** comparing asynchronous implementation (A) to a
  synchronized benchmark (S). The synchronized benchmark sets `α_A = α_B`
  while keeping the same `I_t` path.

- `src/figB1.py`  
  Reproduces **Appendix Figure B1** (phase-style visualization).  
  Uses the same **A** scenario as Figure 2 (logistic `I_t`, `T=120`).  
  Horizontal axis is `G_t = 1 − I_t`. Plots include `π_{R,t}` and `F̄_t`.
  Axis direction (including any inversion) must match the code.

- `src/figC1.py`  
  Reproduces **Appendix Figure C1** (robustness to alternative interoperability
  paths). The transition window is always `t ∈ [T_core, T_conv)` and `ΔF̄` is
  computed on the same window.

## Notation (fixed)

All scripts use the following notation consistently:
- `Y_t` : number of successful transactions
- `F̄_t` : average harm severity per successful transaction
- `H_t = Y_t F̄_t` : total harm

(Do not introduce alternative symbols.)

## Requirements

The code was tested with Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run scripts from the repository root.

(Recommended) create output directories:
```bash
mkdir -p outputs/figures outputs/tables
```

Generate figures:
```bash
python src/fig2.py
python src/fig3.py
python src/fig4.py
python src/figB1.py
python src/figC1.py
```

Each script should save its figure(s) under outputs/figures/.

## Notes on reproducibility

Results may vary slightly across runs if randomness is used.

For strict reproducibility, set and report a fixed random seed inside each
script (or centralize it in a shared config module) and print the seed and
key parameters to stdout.

The code is intended to illustrate qualitative mechanisms (selection under a
sunrise gap) rather than to provide a precise quantitative calibration.