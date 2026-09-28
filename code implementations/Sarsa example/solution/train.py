"""
Train a Sarsa agent on the 5x5 GridWorld and reproduce the two figures from
the notes:
    1. Grid policy + optimal trajectory (mirrors Ch7_Sarsa_example.png)
    2. Total rewards + episode length over training (mirrors
       Ch7_Sarsa-exaple_reward_length_figure.png)

Run with:
    python train.py
"""

from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from agent import SarsaAgent
from env import ACTIONS, GridWorld


# Index of the "stay" action, looked up by value so it stays correct even if
# the action order in env.py is ever changed.
STAY_ACTION = ACTIONS.index((0, 0))


# --- Hyperparameters ----------------------------------------------------------
# Everything below marked `# 可调` is for you to play with.
# See README §四 for the experiments that change these values.

NUM_EPISODES = 500  # 可调
ALPHA = 0.1  # learning rate                              # 可调
GAMMA = 0.9  # discount factor                           # 可调
EPSILON = 0.1  # epsilon-greedy exploration                # 可调
SEED = 0  # 可调


# --- Training -----------------------------------------------------------------
def train(num_episodes=NUM_EPISODES):
    env = GridWorld()
    agent = SarsaAgent(
        num_states=env.num_states(),
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        seed=SEED,
    )

    rewards_history = np.zeros(num_episodes)
    length_history = np.zeros(num_episodes, dtype=int)

    for ep in range(num_episodes):
        total_r, length, _traj = agent.run_episode(env)
        rewards_history[ep] = total_r
        length_history[ep] = length
        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep + 1:>4d}  "
                f"avg_reward(last50)={rewards_history[max(0, ep - 49) : ep + 1].mean():+.3f}  "
                f"avg_length(last50)={length_history[max(0, ep - 49) : ep + 1].mean():6.1f}"
            )

    # Put the env back to start so debugging env.agent_state after training
    # is not confusing.
    env.reset()

    return env, agent, rewards_history, length_history


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
    ax.xaxis.tick_top()  # move both ticks and tick labels to top
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
        # columns labeled at the TOP of the grid (y > rows-0.5 because
        # invert_yaxis makes y grow downward in data coords)
        ax.text(c, rows - 0.25, str(c + 1), size=10, ha="center", va="center")
    for r in range(rows):
        ax.text(-0.75, r, str(r + 1), size=10, ha="center", va="center")

    # forbidden cells (orange)
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

    # target cell (blue)
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
    """Draw a policy arrow in every non-special cell, pointing to the greedy action."""
    rows, cols = env.env_size
    for x in range(cols):
        for y in range(rows):
            if (x, y) == env.target_state or (x, y) in env.forbidden_states:
                continue
            s = env.state_to_index((x, y))
            a = int(np.argmax(Q[s]))
            dx, dy = ACTIONS[a]
            # scale arrow so it stays inside the cell even at the boundary
            scale = 0.3
            color = (0.4660, 0.6740, 0.1880)
            if a == STAY_ACTION:  # 'stay' -> draw a dot
                ax.plot(x, y, marker="o", color=color, markersize=6)
            else:
                ax.annotate(
                    "",
                    xy=(x + dx * scale, y + dy * scale),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                )


def _draw_trajectory(ax, traj):
    """Connect each (state, action) in `traj` with a line on the grid.

    Each entry of `traj` is a (state_tuple, action) pair; coordinates are
    shifted by +0.5 so the line sits between cell centers.
    """
    if not traj:
        return
    xs = [s[0] + 0.5 for s, _ in traj]
    ys = [s[1] + 0.5 for s, _ in traj]
    ax.plot(
        xs, ys, color=(0, 0, 1), linewidth=2, marker="o", markersize=4, markerfacecolor=(0, 0, 1)
    )


def _greedy_trajectory(env: GridWorld, Q: np.ndarray, max_steps=200):
    """Roll out the greedy policy from env.start_state and return the visited (s,a).

    Uses a fresh GridWorld copy so we never mutate the training env's state.
    """
    roll_env = GridWorld(
        env_size=env.env_size,
        start_state=env.start_state,
        target_state=env.target_state,
        forbidden_states=env.forbidden_states,
        reward_target=env.reward_target,
        reward_forbidden=env.reward_forbidden,
        reward_step=env.reward_step,
        actions=env.actions,
        max_steps=env.max_steps,
    )
    s_tuple, _ = roll_env.reset()
    traj = [(s_tuple, None)]
    for _ in range(max_steps):
        s = roll_env.state_to_index(s_tuple)
        a = int(np.argmax(Q[s]))
        next_tuple, _, done, _ = roll_env.step(a)
        traj.append((next_tuple, a))
        if done:
            break
        s_tuple = next_tuple
    return traj


def plot_policy_and_trajectory(env: GridWorld, agent: SarsaAgent):
    fig, ax = plt.subplots(figsize=(5, 5))
    _grid_ax(ax, env)
    _draw_policy_arrows(ax, env, agent.Q)
    traj = _greedy_trajectory(env, agent.Q)
    _draw_trajectory(ax, traj)
    plt.tight_layout()
    return fig


def plot_training_curves(rewards, lengths):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax1.plot(rewards)
    ax1.set_ylabel("Total rewards")
    ax1.axhline(0, color="k", linewidth=0.5, alpha=0.4)

    ax2.plot(lengths)
    ax2.set_ylabel("Episode length")
    ax2.set_xlabel("Episode index")

    plt.tight_layout()
    return fig


# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":
    env, agent, rewards, lengths = train()

    fig1 = plot_policy_and_trajectory(env, agent)
    fig2 = plot_training_curves(rewards, lengths)
    plt.show()
