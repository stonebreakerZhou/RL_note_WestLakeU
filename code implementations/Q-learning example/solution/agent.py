"""
Q-learning (off-policy TD control) agent for the GridWorld example.

Algorithm recap (from
RL_note/RL_note_WestLakeU/Sean's notes/RL_WLU_note.typ
§"5. TD Learning of Optimal Action Values : Q-learning"):

    q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - alpha_t(s_t, a_t)
                         * [ q_t(s_t, a_t) - ( r_{t+1} + gamma * max_a q_t(s_{t+1}, a) ) ]

This file has ONE TODO block for you to fill in:
    TODO 1: Q-learning TD update -- the core formula
            (note: select_action and run_episode are provided directly,
             because the learning here is in the ONE-line formula change
             from Sarsa -- not in episode structure)

Compare to Sarsa example's agent.py to feel the difference.
"""

from __future__ import annotations

import numpy as np

from env import NUM_ACTIONS


class QLearningAgent:
    """Tabular Q-learning with a pluggable behavior policy."""

    def __init__(
        self,
        num_states: int,
        num_actions: int = NUM_ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.9,
        seed: int | None = 0,
        behavior_policy: str = "uniform",  # ← 可调: "uniform" / "biased"
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.behavior_policy = behavior_policy

        # action-value table Q(s, a), initialized to zero
        self.Q = np.zeros((num_states, num_actions), dtype=float)

        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int) -> int:
        """Pick an action using self.behavior_policy.

        Two options are wired in:
          - "uniform" : every action with equal probability 0.2.
          - "biased"  : Zhao-style exploration that leans toward the target.
                        Probabilities are [0.4, 0.4, 0.1, 0.05, 0.05] for
                        actions [down, right, up, left, stay]. Looks "smarter
                        than uniform" but the bias means some side cells
                        stay under-sampled -- a clean test for Q-learning's
                        robustness.
        """
        if self.behavior_policy == "uniform":
            return int(self.rng.integers(self.num_actions))
        if self.behavior_policy == "biased":
            # Zhao-style biased exploration: heavy mass on actions that move
            # TOWARD the target. From start (0,0) the target (2,3) is to the
            # right (action 1) and down (action 0), so those get 0.4 each.
            # The other three (up, left, stay) split the remaining 0.2.
            # Looks "smarter than uniform" but the bias means some side cells
            # stay under-sampled -- good test for Q-learning's robustness.
            p = [0.4, 0.4, 0.1, 0.05, 0.05]  # down, right, up, left, stay
            r = self.rng.random()
            cum = 0.0
            for a, prob in enumerate(p):
                cum += prob
                if r < cum:
                    return a
            return len(p) - 1  # numerical fallback
        raise ValueError(f"Unknown behavior_policy: {self.behavior_policy!r}")

    # =========================================================================
    # TODO 1: Q-learning TD update -- the core formula
    # -------------------------------------------------------------------------
    # Goal: given one transition (s, a, r, s_next, a_next), update Q[s, a]
    #       in place using the Q-learning update rule.
    #
    # Available instance attributes you should use:
    #   - self.alpha  : float, learning rate (e.g. 0.1)
    #   - self.gamma  : float, discount factor (e.g. 0.9)
    #   - self.Q      : np.ndarray of shape (num_states, num_actions)
    #                    self.Q[s_next] is a length-5 array; you want the max.
    #
    # Available numpy functions:
    #   - np.max(a)         : the maximum value in an array
    #   - np.argmax(a)      : the *index* of the maximum (not what we want)
    #
    # Notes:
    #   - The three lines td_target / td_error / Q[s,a] += are the SAME
    #     skeleton as Sarsa; only the TD-target term changes.
    # =========================================================================
    def learn(self, s: int, a: int, r: float, s_next: int, a_next: int) -> None:
        # 注意 Q-learning 并没有用到 a_next
        # === TODO 1 ===
        # Replace the `None` below
        # Until you do, running the code raises
        #     TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
        td_target = r + self.gamma * np.max(self.Q[s_next])  # <-- TODO 1: replace None
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def run_episode(self, env) -> tuple[float, int, list]:
        """Run one episode using self.select_action (the behavior policy) at every step.

        Almost identical to SarsaAgent.run_episode; the ONLY difference is
        that this calls self.learn(...) which uses max instead of a_next.

        Provided directly so you can focus on the one-line TD update.
        """
        state, _ = env.reset()
        state_index = env.state_to_index(state)
        action = self.select_action(state_index)
        traj = [(state, action)]
        total_reward = 0.0
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_state_index = env.state_to_index(next_state)
            next_action = self.select_action(next_state_index)
            self.learn(state_index, action, reward, next_state_index, next_action)
            state, state_index, action = next_state, next_state_index, next_action
            traj.append((state, action))
            total_reward += reward
        return total_reward, len(traj) - 1, traj
