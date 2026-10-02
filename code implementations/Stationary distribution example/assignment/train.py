"""
Stationary-distribution convergence demo on a 2x2 grid.

Given a *fixed* policy pi(a|s), simulate one long trajectory starting from
one cell and plot the running percentage of time spent in each of the 4
states. As the number of steps grows, these percentages converge to the
stationary distribution d_pi(s) of the Markov chain induced by pi.

This is the visualisation for the concept introduced in
"Sean's notes/RL_WLU_note.typ" line 4132:

    "Stationary distribution ... describes the long-run behavior of a
     Markov process. d_pi(s) >= 0 and sum_s d_pi(s) = 1."

To explore different policies: edit the PI matrix below and re-run.
Each ROW of PI is a state; each COLUMN is an action
(down, right, up, left, stay). Each row must sum to 1.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# --- Grid layout -------------------------------------------------------------
#   (0,0) | (1,0)
#   ------+------
#   (0,1) | (1,1)
# Indices: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1).
STATES: list[tuple[int, int]] = [(0, 0), (1, 0), (0, 1), (1, 1)]
N_STATES = len(STATES)
GRID_W, GRID_H = 2, 2

# Actions: down, right, up, left, stay (matches Sarsa/Q-learning examples).
ACTIONS: list[tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
N_ACTIONS = len(ACTIONS)


# --- Policy (EDIT ME) -------------------------------------------------------
# pi(a | s). Each row = a state, each column = an action. Each row sums to 1.
# Default: a deliberately *non-uniform* policy so the four curves converge
# to visibly different levels (NOT all 25%).
PI: np.ndarray = np.array(
    [
        # state 0 (0,0): mostly stay + drift right
        [0.40, 0.10, 0.10, 0.10, 0.30],
        # state 1 (1,0): down + left + stay
        [0.50, 0.10, 0.10, 0.10, 0.20],
        # state 2 (0,1): mostly stay + drift right
        [0.10, 0.50, 0.10, 0.20, 0.10],
        # state 3 (1,1): up + left + stay
        [0.10, 0.20, 0.30, 0.20, 0.20],
    ],
    dtype=float,
)

START_STATE: int = 0  # (0,0); can be 0/1/2/3
NUM_STEPS: int = 10000  # trajectory length
SEED: int = 0  # 可调


# --- Markov chain ------------------------------------------------------------
def build_transition_table() -> np.ndarray:
    """T[s, a] = next state index for deterministic 2x2 grid.

    Out-of-bounds actions *bump* the agent (it stays in place), matching
    env.py in the other examples. So all transitions are well-defined
    on the 4 states.
    """
    pos_of = {xy: i for i, xy in enumerate(STATES)}
    T = np.zeros((N_STATES, N_ACTIONS), dtype=int)
    for s_idx, (x, y) in enumerate(STATES):
        for a_idx, (dx, dy) in enumerate(ACTIONS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                T[s_idx, a_idx] = pos_of[(nx, ny)]
            else:
                T[s_idx, a_idx] = s_idx  # bump
    return T


def induced_chain(PI: np.ndarray, T: np.ndarray) -> np.ndarray:
    """P_pi[s, s'] = sum_a pi(a|s) * [T[s, a] == s'] -- the transition
    matrix of the Markov chain you get by following pi on the grid."""
    P_pi = np.zeros((N_STATES, N_STATES), dtype=float)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            P_pi[s, T[s, a]] += PI[s, a]
    return P_pi


def stationary_distribution(P_pi: np.ndarray) -> np.ndarray:
    """Solve d_pi = d_pi P_pi, i.e. left eigenvector of P_pi with eigenvalue 1.

    We do this *directly* (not via simulation) so the empirical curves
    have a clean theoretical target to compare against.
    """
    # d_pi (row) satisfies d_pi = d_pi P_pi  <=>  d_pi^T = P_pi^T d_pi^T.
    eigvals, eigvecs = np.linalg.eig(P_pi.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    d_pi = np.real(eigvecs[:, idx])
    # Numerical noise can give tiny negative entries; clip and renormalise.
    d_pi = np.clip(d_pi, 0.0, None)
    d_pi = d_pi / d_pi.sum()
    return d_pi


def simulate(
    PI: np.ndarray,
    T: np.ndarray,
    start_state: int,
    num_steps: int,
    seed: int,
) -> np.ndarray:
    """One trajectory. Returns percentages[t, s] = fraction of steps 0..t
    spent at state s (length num_steps+1, includes the starting state)."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(N_STATES, dtype=int)
    history = np.zeros((num_steps + 1, N_STATES), dtype=float)

    s = start_state
    for t in range(num_steps + 1):
        counts[s] += 1
        history[t] = counts / counts.sum()
        a = int(rng.choice(N_ACTIONS, p=PI[s]))
        s = T[s, a]

    return history


# --- Plot --------------------------------------------------------------------
_STATE_COLORS = ["C0", "C1", "C2", "C3"]


def plot_convergence(
    history: np.ndarray,
    num_steps: int,
    start_state: int,
) -> "tuple[plt.Figure, plt.Axes]":
    """Four empirical curves = one per state. Theoretical d_pi is printed
    in the terminal, not drawn here, to keep the plot uncluttered."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in range(N_STATES):
        ax.plot(
            range(num_steps + 1),
            history[:, s],
            color=_STATE_COLORS[s],
            lw=1.8,
            label=str(STATES[s]),
        )
    ax.set_xlabel("Step index")
    ax.set_ylabel("Percentage of each state visited")
    ax.set_xlim(0, num_steps)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Convergence to stationary distribution  (start = {STATES[start_state]})")
    ax.legend(loc="center right", fontsize=10, framealpha=0.9, title="State")
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig, ax


# --- Entry point -------------------------------------------------------------
if __name__ == "__main__":
    # Validate the policy before running.
    row_sums = PI.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Each row of PI must sum to 1, got {row_sums.tolist()}"
    assert PI.shape == (N_STATES, N_ACTIONS), (
        f"PI must be shape ({N_STATES}, {N_ACTIONS}), got {PI.shape}"
    )

    T = build_transition_table()
    P_pi = induced_chain(PI, T)
    d_pi = stationary_distribution(P_pi)

    print("Stationary distribution d_π (theoretical):")
    for s, xy in enumerate(STATES):
        print(f"  {xy}:  {d_pi[s]:.4f}")
    print(f"  sum = {d_pi.sum():.4f}")

    history = simulate(PI, T, START_STATE, NUM_STEPS, SEED)
    print(f"\nEmpirical fraction after {NUM_STEPS} steps from {STATES[START_STATE]}:")
    for s, xy in enumerate(STATES):
        print(f"  {xy}:  {history[-1, s]:.4f}")

    fig, _ = plot_convergence(history, NUM_STEPS, START_STATE)
    plt.show()
