import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

OBS_MODE = "semantic6"
REWARD_MODE = "balanced"
GRID_SIZE = 10

_COLORS = np.array([
    [0, 0, 0],         # 0 = BLACK
    [255, 255, 255],   # 1 = WHITE
    [101, 67, 33],     # 2 = BROWN
    [160, 161, 161],   # 3 = GREY
    [31, 198, 0],      # 4 = GREEN
    [255, 0, 0],       # 5 = RED
    [255, 127, 127],   # 6 = LIGHT_RED
], dtype=np.uint8)

# Names for readability
BLACK, WHITE, BROWN, GREY, GREEN, RED, LIGHT_RED = _COLORS


def _mask(grid: np.ndarray, color: np.ndarray) -> np.ndarray:
    """
    Build a binary mask where cells matching the color are 1.0.
    """
    return np.all(grid == color, axis=2).astype(np.float32)


def _encode_grid(grid: np.ndarray) -> np.ndarray:
    """
    Encode RGB grid cells using color IDs from _COLORS.
    """
    h, w = grid.shape[:2]
    flat = grid.reshape(h * w, 3)
    encoded = np.zeros(h * w, dtype=np.int32)
    for color_id, color in enumerate(_COLORS):
        encoded[np.all(flat == color, axis=1)] = color_id
    return encoded


def _observation_compact105(grid: np.ndarray) -> np.ndarray:
    """
    YusufAq1 observation function
    """
    h, w = grid.shape[:2]
    encoded = _encode_grid(grid)
    normalized = encoded.astype(np.float32) / 6.0

    agent_idxs = np.where(encoded == 3)[0]
    if len(agent_idxs) > 0:
        idx = int(agent_idxs[0])
        row = idx // w
        col = idx % w
        agent_row = row / max(h - 1, 1)
        agent_col = col / max(w - 1, 1)

        adjacent_vals = []
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n_row = row + d_row
            n_col = col + d_col
            if 0 <= n_row < h and 0 <= n_col < w:
                adjacent_vals.append(encoded[n_row * w + n_col])
        adjacent_danger = sum(1 for value in adjacent_vals if value in (5, 6)) / max(len(adjacent_vals), 1)
    else:
        agent_row = 0.0
        agent_col = 0.0
        adjacent_danger = 0.0

    walls = int(np.sum(encoded == 2))
    total_coverable = max(h * w - walls, 1)
    explored = int(np.sum(encoded == 1) + np.sum(encoded == 3) + np.sum(encoded == 6))
    coverage_frac = explored / total_coverable

    surveillance = int(np.sum((encoded == 5) | (encoded == 6)))
    global_danger = surveillance / total_coverable

    return np.concatenate(
        [
            normalized,
            np.array([agent_row, agent_col, coverage_frac, adjacent_danger, global_danger], dtype=np.float32),
        ]
    ).astype(np.float32)


def _observation_semantic6(grid: np.ndarray) -> np.ndarray:
    """
    Daniel observation function
    """
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

    if OBS_MODE == "compact105":
        return gym.spaces.Box(low=0.0, high=1.0, shape=(105,), dtype=np.float32)

    raise ValueError(f"Unsupported OBS_MODE: {OBS_MODE}")


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    if OBS_MODE == "default":
        return grid.flatten()

    if OBS_MODE == "semantic6":
        return _observation_semantic6(grid)

    if OBS_MODE == "compact105":
        return _observation_compact105(grid)

    raise ValueError(f"Unsupported OBS_MODE: {OBS_MODE}")


def _reward_balanced(info: dict) -> float:
    """
    Daniel reward function
    """
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


def _reward_main_risk(info: dict) -> float:
    """
    YusufAq1 reward function
    """
    if info["game_over"]:
        return -50.0

    if info["cells_remaining"] == 0:
        efficiency = info["steps_remaining"] / 500.0
        return 100.0 + efficiency * 50.0

    agent_pos = info["agent_pos"]
    all_fov_flat = {
        row * GRID_SIZE + col
        for enemy in info["enemies"]
        for row, col in enemy.get_fov_cells()
    }
    adjacent_flat = {agent_pos - GRID_SIZE, agent_pos + GRID_SIZE, agent_pos - 1, agent_pos + 1}
    adjacent_danger = len(adjacent_flat & all_fov_flat)

    reward_value = 1.0 if info["new_cell_covered"] else -0.05
    reward_value -= adjacent_danger * 0.2
    return reward_value


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
    if REWARD_MODE == "main_risk":
        return float(_reward_main_risk(info))

    raise ValueError("Unsupported REWARD_MODE: "
                     f"{REWARD_MODE}. Available modes: balanced, main_risk")
