"""
Train a Q-learning agent on the 5x5 GridWorld and reproduce the THREE figures
from Zhao's Q-learning lecture slides
(Sean's notes/images/Ch7_Qlearning_eg_uniform_pi_b.png):

    1. The generated episode (experience) of π_b:
       ONE episode of NUM_TRAINING_STEPS steps, starting from the top-left
       start cell. Each cell is filled with a heatmap colour showing how many
       time steps the agent spent in it (the agent moves from cell centre to
       cell centre; the number at each centre is the visit count).

    2. The state value error of Q-learning vs step in the episode:
       For every step we plot sum_s | max_a Q(s, a) - V*(s) |, where V* is the
       *true* optimal state value obtained by solving Bellman optimality
       via value iteration (NOT via the Q-learning agent itself).

    3. The policy found by off-policy Q-learning:
       The full-grid arrow matrix (one arrow per non-special cell pointing
       to argmax_a Q[s, a]). No single trajectory is drawn, because Q*
       applies to every state, not just the start state -- drawing a
       trajectory from (0,0) would be a Sarsa-style artefact.

Switch BEHAVIOR_POLICY below to compare uniform / biased.
Run with:
    python train.py
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent
from env import ACTIONS, GridWorld


# Index of the "stay" action, looked up by value so it stays correct even if
# the action order in env.py is ever changed.
STAY_ACTION = ACTIONS.index((0, 0))


# --- Hyperparameters ----------------------------------------------------------
# Everything below marked `# 可调` is for you to play with.
# See README §四 for the experiments that change these values.

# Total env steps to train for. Zhao's slides use 1e5 (100_000) steps; we
# default to 10_000 because uniform π_b is dense enough to fill the grid
# at this scale and it runs in well under a second. Set to 100_000 to match
# the slides exactly.
NUM_TRAINING_STEPS = 100_000  # 可调
ALPHA = 0.1  # learning rate                              # 可调
GAMMA = 0.9  # discount factor                            # 可调
SEED = 0  # 可调
# Q-learning only: which behavior policy to use during training
# (uniform = each action with prob 1/5; biased = 0.4 right + 0.4 down +
#           0.1 up + 0.05 left + 0.05 stay, Zhao-style: looks like it heads
#           toward target but under-samples some side cells).
BEHAVIOR_POLICY = "uniform"  # 可调: "uniform" / "biased"


# --- Ground-truth V* ---------------------------------------------------------
def value_iteration(env: GridWorld, gamma: float = 0.9, n_iter: int = 300) -> np.ndarray:
    """Compute the optimal state value V* by solving Bellman optimality directly.

    For every state s (the target included -- the episode does NOT end there,
    staying at the target earns +1 every step, so V*(target) = 1/(1-gamma)):
        V*(s) = max_a [ r(s, a) + gamma * V*(s') ]

    We use this as ground truth to measure |max_a Q(s, a) - V*(s)|. Solving it
    with value iteration (not via Q-learning itself) keeps the comparison
    honest -- V* is independent of how the agent explored.
    """
    n = env.num_states()
    cols = env.env_size[1]
    V = np.zeros(n)

    def idx(x: int, y: int) -> int:
        return x * cols + y

    def reward_and_next(x: int, y: int, a: int) -> tuple[float, int]:
        """Mirror env._transition: return (reward, next_state_index).

        Bumping into a forbidden cell or the grid boundary keeps the agent
        in place and gives the forbidden reward -- same as in env.py.
        """
        dx, dy = env.actions[a]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < env.env_size[0] and 0 <= ny < env.env_size[1]):
            return env.reward_forbidden, idx(x, y)
        if (nx, ny) in env.forbidden_states:
            return env.reward_forbidden, idx(x, y)
        if (nx, ny) == env.target_state:
            return env.reward_target, idx(nx, ny)
        return env.reward_step, idx(nx, ny)

    for _ in range(n_iter):
        V_new = V.copy()
        for s in range(n):
            x, y = divmod(s, cols)
            qs = np.empty(env.num_actions)
            for a in range(env.num_actions):
                r, s_next = reward_and_next(x, y, a)
                qs[a] = r + gamma * V[s_next]
            V_new[s] = qs.max()
        V = V_new
    return V


# --- Training ----------------------------------------------------------------
def train(num_steps: int = NUM_TRAINING_STEPS) -> tuple[GridWorld, QLearningAgent, dict]:
    """Generate ONE episode of `num_steps` steps with π_b from the start cell,
    and let Q-learning learn from it step by step (off-policy: π_b only
    produces the experience, the TD target uses max_a Q[s_next, a]).

    The episode is NOT terminated when the target is reached -- the agent just
    keeps walking (as in Zhao's slides), so the whole experience is a single
    continuous trace of length `num_steps`.

    Returns (env, agent, log) where log contains:
        - 'visited': list of (x, y), the state s_t the agent occupies at each
                     step t = 0 .. num_steps-1 (s_0 = start cell).
        - 'state_value_error': sum over non-forbidden s of
                               |max_a Q(s, a) - V*(s)|, after each update.
    """
    env = GridWorld()
    # Override the per-episode step cap so env.step() never returns done=True
    # for hitting max_steps -- we want one continuous episode.
    env.max_steps = num_steps + 1

    agent = QLearningAgent(
        num_states=env.num_states(),
        alpha=ALPHA,
        gamma=GAMMA,
        seed=SEED,
        behavior_policy=BEHAVIOR_POLICY,
    )

    # ground-truth V* for state value error
    V_star = value_iteration(env, gamma=GAMMA)

    # Forbidden cells can never be occupied (env.py bumps the agent back),
    # so they are left out of the error sum.
    reachable = np.array(
        [divmod(s, env.env_size[1]) not in env.forbidden_states for s in range(env.num_states())]
    )

    state, _ = env.reset()
    visited: list[tuple[int, int]] = []
    errors: list[float] = []

    for _ in range(num_steps):
        visited.append(state)
        s = env.state_to_index(state)
        a = agent.select_action(s)
        # `done` is ignored on purpose: reaching the target does not end the
        # episode, the agent keeps walking from there.
        next_state, r, _done, _ = env.step(a)
        s_next = env.state_to_index(next_state)
        a_next = agent.select_action(s_next)
        agent.learn(s, a, r, s_next, a_next)

        errors.append(float(np.sum(np.abs(agent.Q.max(axis=1) - V_star)[reachable])))
        state = next_state

    env.reset()
    return (
        env,
        agent,
        {
            "visited": visited,
            "state_value_error": errors,
        },
    )


# --- Visualization ------------------------------------------------------------
def _grid_ax(ax, env: GridWorld):
    """Draw the colored grid (target + forbidden + start) on `ax`."""
    rows, cols = env.env_size
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(True, linestyle="-", color="gray", linewidth=1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.tick_params(
        bottom=False,
        left=False,
        right=False,
        top=False,
        labelbottom=False,
        labelleft=False,
        labeltop=False,
    )
    for c in range(cols):
        ax.text(c, rows - 0.25, str(c + 1), size=10, ha="center", va="center")
    for r in range(rows):
        ax.text(-0.75, r, str(r + 1), size=10, ha="center", va="center")

    for x, y in env.forbidden_states:
        ax.add_patch(
            patches.Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=(0.9290, 0.6940, 0.125),
                edgecolor="gray",
                linewidth=1,
            )
        )

    tx, ty = env.target_state
    ax.add_patch(
        patches.Rectangle(
            (tx - 0.5, ty - 0.5),
            1,
            1,
            facecolor=(0.3010, 0.7450, 0.9330),
            edgecolor="gray",
            linewidth=1,
        )
    )


def _draw_policy_arrows(ax, env: GridWorld, Q: np.ndarray):
    """Draw the greedy action argmax_a Q[s, a] in every non-forbidden cell
    (target included): an arrow for a move, a dot for "stay".
    """
    rows, cols = env.env_size
    color = (0.4660, 0.6740, 0.1880)
    scale = 0.3

    for x in range(cols):
        for y in range(rows):
            if (x, y) in env.forbidden_states:
                continue
            s = env.state_to_index((x, y))
            a = int(np.argmax(Q[s]))
            dx, dy = ACTIONS[a]
            if a == STAY_ACTION:
                ax.plot(x, y, marker="o", color=color, markersize=6)
            else:
                ax.annotate(
                    "",
                    xy=(x + dx * scale, y + dy * scale),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                )


def plot_generated_episode(env: GridWorld, visited: list):
    """Zhao's "generated episode", shown as a visit-count heatmap.

    Coordinates: the centre of cell (x, y) is the point (x, y) and the cell
    spans [x-0.5, x+0.5] x [y-0.5, y+0.5] (grid lines sit on the half
    integers). The agent moves from centre to centre, so every cell is filled
    completely with a colour for its visit count, and the count is written at
    its centre.

    Forbidden cells keep their orange fill (they can never be entered, count
    = 0); the target keeps a thick blue border so it stays recognisable.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    _grid_ax(ax, env)

    rows, cols = env.env_size
    counts = np.zeros((cols, rows), dtype=int)
    for x, y in visited:
        counts[x, y] += 1

    cmap = plt.cm.YlGn
    norm = mcolors.Normalize(vmin=0, vmax=max(int(counts.max()), 1))

    for x in range(cols):
        for y in range(rows):
            if (x, y) in env.forbidden_states:
                continue
            n = int(counts[x, y])
            ax.add_patch(
                patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor=cmap(norm(n)),
                    edgecolor="gray",
                    linewidth=1,
                    zorder=2,
                )
            )
            ax.text(
                x,
                y,
                str(n),
                fontsize=9,
                ha="center",
                va="center",
                zorder=4,
                color="white" if norm(n) > 0.6 else "black",
            )

    tx, ty = env.target_state
    ax.add_patch(
        patches.Rectangle(
            (tx - 0.5, ty - 0.5),
            1,
            1,
            fill=False,
            edgecolor=(0.3010, 0.7450, 0.9330),
            linewidth=3,
            zorder=3,
        )
    )

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Visit count")
    ax.set_title(f"Generated episode ({len(visited)} steps, π_b = {BEHAVIOR_POLICY})")
    plt.tight_layout()
    return fig


def plot_state_value_error(errors: list, num_steps: int):
    """Zhao's figure (b): state value error of Q-learning vs step.

    After each step we record sum_s | max_a Q(s, a) - V*(s) | over all
    non-forbidden states. V* is the true optimal state value (computed by
    value iteration, independent of the agent). The error starts high (Q is
    initialized to zero, far from V*) and decays towards 0 as the episode
    keeps covering the grid.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(errors)), errors)
    ax.set_xlabel("Step in the episode")
    ax.set_ylabel("State error")
    ax.set_xlim(0, num_steps)
    ax.set_title(f"Q-learning result (π_b = {BEHAVIOR_POLICY}, α = {ALPHA}, γ = {GAMMA})")
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_estimated_policy(env: GridWorld, agent: QLearningAgent):
    """Zhao's figure (a): the policy found by off-policy Q-learning.

    A full-grid arrow matrix -- one arrow per non-special cell pointing to
    argmax_a Q[s, a]. No trajectory is drawn, because Q* applies to every
    state: drawing a single trajectory from (0,0) would be a Sarsa-style
    artefact that hides the fact that Q* is a complete grid-wide optimum.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    _grid_ax(ax, env)
    _draw_policy_arrows(ax, env, agent.Q)
    ax.set_title(f"Estimated policy (π_b = {BEHAVIOR_POLICY})")
    plt.tight_layout()
    return fig


# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":
    env, agent, log = train()

    print(f"\nGenerated episode: {len(log['visited'])} steps, pi_b = {BEHAVIOR_POLICY!r}")
    if log["state_value_error"]:
        first, last = log["state_value_error"][0], log["state_value_error"][-1]
        print(f"State value error: first = {first:.4f}, last = {last:.4f}")

    fig1 = plot_generated_episode(env, log["visited"])
    fig2 = plot_state_value_error(log["state_value_error"], len(log["visited"]))
    fig3 = plot_estimated_policy(env, agent)
    plt.show()
