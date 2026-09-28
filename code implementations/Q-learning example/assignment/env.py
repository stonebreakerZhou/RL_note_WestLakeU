"""
GridWorld environment shared by the Sarsa and Q-learning examples
(the layout matches the figure in
RL_note/RL_note_WestLakeU/Sean's notes/RL_WLU_note.typ §"4) Sarsa —— Example").

This file is intentionally the same as the one in the Sarsa example, so the
two algorithms can be compared under identical conditions.

Layout (5x5, row=y, col=x, both 0-indexed):
    - start  = (0, 0)   = col 1, row 1   top-left, marked with '*' in the figure
    - target = (2, 3)   = col 3, row 4   blue square in the figure
    - forbidden = {(1,1),(2,1),(2,2),(1,3),(3,3),(1,4)}
                        = (col2,row2) (col3,row2) (col3,row3)
                          (col2,row4) (col4,row4) (col2,row5)
                        the orange cells in the figure -- they form a wall that
                        forces the agent to detour via the right edge
    - reward: target=+1, forbidden=-1, step=0
    - actions: down, right, up, left, stay   (5 actions)
    - max_steps = 150 per episode (see note below)

Why the 150-step cap: an untrained agent can wander for thousands of steps.
Without a cap the very first episode runs thousands of steps, which updates Q
thousands of times and makes the agent "converge" in a single episode -- you
then see a flat curve and learn nothing about how TD learning gradually
improves. The cap lives in the environment, so run_episode() needs no extra
logic: env.step() simply reports done=True once the horizon is reached.

The optimal path is 11 steps:
    (0,0) ->R-> (1,0) ->R-> (2,0) ->R-> (3,0) ->R-> (4,0)
          ->D-> (4,1) ->D-> (4,2) ->D-> (4,3) ->D-> (4,4)
          ->L-> (3,4) ->L-> (2,4) ->U-> (2,3) = target
which matches the blue line drawn in the notes' figure.

The environment intentionally matches the lecture notes, NOT the defaults in
"Code for grid world/python_version/src/arguments.py".
"""

from __future__ import annotations


# --- Environment configuration ------------------------------------------------
# Everything below marked `# 可调` is for you to play with.
# See README §四 for the experiments that change these values.

ENV_SIZE = (5, 5)  # (rows, cols)        # 可调
START_STATE = (0, 0)  # 可调
TARGET_STATE = (2, 3)  # 可调
FORBIDDEN_STATES = {(1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)}  # 可调

REWARD_TARGET = 1.0  # 可调
REWARD_FORBIDDEN = -1.0  # 可调
REWARD_STEP = 0.0  # 可调

MAX_STEPS_PER_EPISODE = 150  # 可调

# (dx, dy): down, right, up, left, stay
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]  # 可调
NUM_ACTIONS = len(ACTIONS)


class GridWorld:
    """A small 5x5 grid world for the Sarsa example.

    State is a tuple (x, y). Actions are integers 0..4 indexing ACTIONS.
    Hitting a forbidden cell or stepping outside the grid gives REWARD_FORBIDDEN
    and the agent stays in place (so the cell becomes a 'wall' you bump into).
    """

    def __init__(
        self,
        env_size=ENV_SIZE,
        start_state=START_STATE,
        target_state=TARGET_STATE,
        forbidden_states=FORBIDDEN_STATES,
        reward_target=REWARD_TARGET,
        reward_forbidden=REWARD_FORBIDDEN,
        reward_step=REWARD_STEP,
        actions=ACTIONS,
        max_steps=MAX_STEPS_PER_EPISODE,
    ):
        self.env_size = env_size
        self.start_state = tuple(start_state)
        self.target_state = tuple(target_state)
        self.forbidden_states = set(map(tuple, forbidden_states))
        self.reward_target = reward_target
        self.reward_forbidden = reward_forbidden
        self.reward_step = reward_step
        self.actions = actions
        self.num_actions = len(actions)
        self.max_steps = max_steps

        self.agent_state = self.start_state
        self.step_count = 0

    # ----- core API ----------------------------------------------------------
    def reset(self):
        """Reset the agent to the start state and the step counter.

        Returns: (state, info)
        """
        self.agent_state = self.start_state
        self.step_count = 0
        return self.agent_state, {}

    def step(self, action: int):
        """Take action `action` (int in [0, num_actions)).

        Returns: (next_state, reward, done, info)

        `done` is True either when the agent reaches the target or when the
        episode has used up `max_steps` steps.
        """
        assert 0 <= action < self.num_actions, f"Invalid action {action}"
        next_state, reward = self._transition(self.agent_state, action)
        self.agent_state = next_state
        self.step_count += 1
        done = (next_state == self.target_state) or (self.step_count >= self.max_steps)
        return next_state, reward, done, {}

    # ----- internals ---------------------------------------------------------
    def _transition(self, state, action):
        x, y = state
        dx, dy = self.actions[action]
        nx, ny = x + dx, y + dy

        # outside the grid -> bump, forbidden reward
        if not (0 <= nx < self.env_size[0] and 0 <= ny < self.env_size[1]):
            return state, self.reward_forbidden

        # stepping onto a forbidden cell -> bump, forbidden reward
        if (nx, ny) in self.forbidden_states:
            return state, self.reward_forbidden

        # reaching the target
        if (nx, ny) == self.target_state:
            return (nx, ny), self.reward_target

        # normal step
        return (nx, ny), self.reward_step

    # ----- helpers used by visualization ------------------------------------
    def num_states(self):
        return self.env_size[0] * self.env_size[1]

    def state_to_index(self, state):
        x, y = state
        return x * self.env_size[1] + y
