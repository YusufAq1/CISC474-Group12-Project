import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

OBS_MODE = "semantic6"
REWARD_MODE = "balanced"
INITIAL_STEPS = 500

# RGB colors from env.py
BLACK = np.array([0, 0, 0], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)
BROWN = np.array([101, 67, 33], dtype=np.uint8)
GREY = np.array([160, 161, 161], dtype=np.uint8)
GREEN = np.array([31, 198, 0], dtype=np.uint8)
RED = np.array([255, 0, 0], dtype=np.uint8)
LIGHT_RED = np.array([255, 127, 127], dtype=np.uint8)


def _mask(grid: np.ndarray, color: np.ndarray) -> np.ndarray:
    """
    Build a binary mask where cells matching the color are 1.0.
    """
    return np.all(grid == color, axis=2).astype(np.float32)


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    if OBS_MODE == "default":
        cell_values = env.grid + 256
        return gym.spaces.MultiDiscrete(cell_values.flatten())

    if OBS_MODE == "semantic6":
        if hasattr(env, "grid") and isinstance(env.grid, np.ndarray) and env.grid.ndim >= 2:
            h, w = env.grid.shape[:2]
        elif hasattr(env, "grid_size"):
            h = int(env.grid_size)
            w = int(env.grid_size)
        else:
            raise ValueError("Cannot infer grid dimensions for semantic6 observation space.")

        return gym.spaces.Box(low=0.0, high=1.0, shape=(6 * h * w,), dtype=np.float32)

    raise ValueError(f"Unsupported OBS_MODE: {OBS_MODE}")


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    if OBS_MODE == "default":
        return grid.flatten()

    if OBS_MODE == "semantic6":
        h, w = grid.shape[:2]

        agent = _mask(grid, GREY)
        wall = _mask(grid, BROWN)
        enemy = _mask(grid, GREEN)

        red = _mask(grid, RED)
        light_red = _mask(grid, LIGHT_RED)
        white = _mask(grid, WHITE)
        black = _mask(grid, BLACK)

        danger_now = np.clip(red + light_red, 0.0, 1.0)
        visited = np.clip(white + light_red + agent, 0.0, 1.0)
        unvisited = np.clip(black + red, 0.0, 1.0)

        channels = np.stack(
            [
                agent,
                wall,
                enemy,
                danger_now,
                visited,
                unvisited,
            ],
            axis=0,
        ).astype(np.float32)

        return channels.reshape(6 * h * w).astype(np.float32)

    raise ValueError(f"Unsupported OBS_MODE: {OBS_MODE}")


def _reward_balanced(info: dict) -> float:
    cells_remaining = info["cells_remaining"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]

    reward_value = 0.0
    reward_value -= 0.01

    if new_cell_covered:
        reward_value += 0.30
    else:
        reward_value -= 0.03

    if game_over:
        reward_value -= 12.0
    elif cells_remaining == 0:
        reward_value += 12.0
    elif steps_remaining == 0:
        reward_value -= 3.0

    return reward_value


def _reward_zero(info: dict) -> float:
    """
    Baseline reward for debug.
    """
    return 0.0


def reward(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    if REWARD_MODE == "balanced":
        return float(_reward_balanced(info))
    if REWARD_MODE == "zero":
        return float(_reward(info))

    raise ValueError("Unsupported REWARD_MODE: "
                     f"{REWARD_MODE}. Available modes: balanced, zero")
