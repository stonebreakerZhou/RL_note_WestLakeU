"""
Sarsa (on-policy TD control) agent for the GridWorld example.

Algorithm recap (from RL_note/RL_note_WestLakeU/RL_WLU_note.typ §"3) TD Learning of
Action Values : Sarsa"):

    q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - alpha_t(s_t, a_t)
                         * [ q_t(s_t, a_t) - ( r_{t+1} + gamma * q_t(s_{t+1}, a_{t+1}) ) ]

This file has THREE TODO blocks for you to fill in:
    TODO 1: epsilon-greedy action selection
    TODO 2: Sarsa TD update
    TODO 3: one episode rollout
"""

from __future__ import annotations

import numpy as np

from env import NUM_ACTIONS


class SarsaAgent:
    """Tabular Sarsa with epsilon-greedy behavior policy."""

    def __init__(
        self,
        num_states: int,
        num_actions: int = NUM_ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int | None = 0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # action-value table Q(s, a), initialized to zero
        self.Q = np.zeros((num_states, num_actions), dtype=float)

        self.rng = np.random.default_rng(seed)

    # =========================================================================
    # TODO 1: finish the epsilon-greedy action selection function
    # -------------------------------------------------------------------------
    # Goal: given a state index, return an action index in [0, num_actions).
    #
    # Algorithm (epsilon-greedy):
    #   1. Roll a uniform random number r in [0, 1).
    #   2. If r < self.epsilon   -> exploration: pick a RANDOM action.
    #      Else                  -> exploitation: pick the action with the
    #                              largest Q value for this state.
    #
    # Available instance attributes you should use:
    #   - self.epsilon            : float, exploration probability
    #   - self.num_actions        : int,   number of actions (5 for this env)
    #   - self.Q                  : np.ndarray of shape (num_states, num_actions)
    #                               self.Q[state] is a length-5 array of Q values
    #                               for the 5 actions at this state
    #   - self.rng                : np.random.Generator, USE THIS instead of the
    #                               global np.random. Methods you'll likely want:
    #                                 self.rng.random()           -> float in [0, 1)
    #                                 self.rng.integers(n)        -> int in [0, n)
    #                               Using self.rng keeps the experiment
    #                               reproducible (same seed -> same rollout).
    #
    # Useful numpy functions:
    #   - np.argmax(a)            : index of the maximum element in a
    #
    # Returns: an int in [0, self.num_actions).
    # =========================================================================
    def select_action(self, state: int) -> int:
        raise NotImplementedError("TODO 1: implement epsilon-greedy action selection")

    # =========================================================================
    # TODO 2: Sarsa TD update (the core formula)
    # -------------------------------------------------------------------------
    # Goal: given one transition (s, a, r, s_next, a_next), update Q[s, a]
    #       in place using the Sarsa update rule.
    #
    # Formula (from the notes' red box):
    #
    #       Q[s, a] += alpha * ( r + gamma * Q[s_next, a_next] - Q[s, a] )
    #
    # Equivalently, broken into 3 lines:
    #
    #       td_target = r + gamma * Q[s_next, a_next]
    #       td_error  = td_target - Q[s, a]      # NOTE the sign: target minus current
    #       Q[s, a]  += alpha * td_error         # in-place update, do NOT replace
    #
    # Available instance attributes you should use:
    #   - self.alpha  : float, learning rate (e.g. 0.1)
    #   - self.gamma  : float, discount factor (e.g. 0.9)
    #   - self.Q      : np.ndarray of shape (num_states, num_actions)
    #
    # Important: this is the *on-policy* form -- a_next is the action that the
    # current behavior policy ACTUALLY takes in s_next, not the greedy one
    # (that's what distinguishes Sarsa from Q-learning).
    #
    # Returns: nothing. Mutates self.Q in place.
    # =========================================================================
    def learn(self, s: int, a: int, r: float, s_next: int, a_next: int) -> None:
        raise NotImplementedError("TODO 2: implement the Sarsa TD update")

    # =========================================================================
    # TODO 3: run one episode using the current behavior policy
    # -------------------------------------------------------------------------
    # Goal: roll out one full episode using the current epsilon-greedy policy,
    #       updating Q along the way. Return (total_reward, length, traj).
    #
    # High-level shape (skeleton -- you fill in the body):
    #
    #     # Phase 1: start the episode
    #     ???                                    # reset env, get first state
    #     ???                                    # convert tuple -> index for Q
    #     ???                                    # choose first action
    #     ???                                    # initialize traj and total_reward
    #
    #     # Phase 1.5: initialize done first
    #     ???                                    # done = False (so while can start)
    #
    #     # Phase 2: take steps until done
    #     while ???:                             # what condition ends the loop?
    #         ???                                # (a) advance the env one step (updates done)
    #         ???                                # (b) convert new state to index
    #         ???                                # (c) choose next action
    #         ???                                # (d) update Q with this transition
    #         ???                                # (e) advance (state, action, index) -- all three!
    #         ???                                # (f) record into traj: traj.append((state, action))
    #         ???                                # (g) accumulate reward
    #
    #     # Phase 3: report results
    #     return ???, ???, ???                   # total_reward, length, traj
    #
    # Tools at your disposal (signatures given above in the file header):
    #   env.reset()                     -> (state_tuple, info)
    #   env.step(action)                -> (next_state_tuple, reward, done, info)
    #   env.state_to_index(state_tuple) -> int        # bridges (x,y) -> Q row
    #   self.select_action(state_index) -> int        # from TODO 1
    #   self.learn(s, a, r, s_next, a_next)           # from TODO 2, mutates self.Q
    #
    # Things to think about as you write:
    #   - State vs state_index: Q is a (num_states, num_actions) array, so you
    #     always need the int index to look up or update Q. But env speaks in
    #     (x, y) tuples. Convert at the boundary, in BOTH directions.
    #   - Order matters: (a) step first, then (c) pick next action, then (d) learn.
    #     Sarsa uses the action the behavior policy WOULD take in s_next --
    #     that's epsilon-greedy (select_action), NOT argmax. So pick next_action
    #     BEFORE learn -- otherwise you are writing Q-learning, not Sarsa.
    #   - Trajectory format: traj is a LIST (not a tuple) of (state_tuple, action)
    #     pairs. Look at _greedy_trajectory in train.py to see what shape it needs.
    #   - traj.append(...) takes EXACTLY ONE argument -- a tuple `(state, action)`,
    #     NOT two arguments. `traj.append(state, action)` will raise
    #     `TypeError: append() takes exactly one argument (2 given)`.
    #   - Done flag: env.step tells you when the episode ends (target reached
    #     OR max_steps hit). Stop the loop when done is true.
    #   - Episode length = number of steps taken, NOT number of entries in
    #     traj. Since traj includes the starting cell, len(traj) - 1 = length.
    #
    # Returns:
    #   - total_reward: float, sum of all rewards collected this episode
    #   - length:       int,   number of env.step() calls (= number of steps)
    #   - traj:         list of (state_tuple, action) pairs, used by train.py
    # =========================================================================
    def run_episode(self, env) -> tuple[float, int, list]:
        raise NotImplementedError("TODO 3: implement one-episode rollout")