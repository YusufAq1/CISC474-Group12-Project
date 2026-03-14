import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

def observation_space(env: gym.Env) -> gym.spaces.Space:
    # 100 normalized grid values + agent_row + agent_col + coverage_fraction
    return gym.spaces.Box(low=0.0, high=1.0, shape=(103,), dtype=np.float32)


def observation(grid: np.ndarray):
    encoded = _encode_grid(grid)                          # shape (100,), int 0-6
    normalized = encoded.astype(np.float32) / 6.0        # normalize to [0, 1]

    # Extract agent position (encoded value 3)
    agent_idxs = np.where(encoded == 3)[0]
    if len(agent_idxs) > 0:
        idx = agent_idxs[0]
        agent_row = (idx // 10) / 9.0
        agent_col = (idx % 10) / 9.0
    else:
        agent_row, agent_col = 0.0, 0.0

    # Compute coverage fraction from grid
    walls = np.sum(encoded == 2)
    total_coverable = max(100 - walls, 1)
    explored = np.sum(encoded == 1) + np.sum(encoded == 3) + np.sum(encoded == 6)
    coverage_frac = float(explored) / total_coverable

    return np.concatenate([normalized, [agent_row, agent_col, coverage_frac]]).astype(np.float32)


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
    if info["game_over"]:
        return -50.0
    if info["cells_remaining"] == 0:
        efficiency = info["steps_remaining"] / 500.0
        return 100.0 + efficiency * 50.0  # up to +150 for fast finish
    if info["new_cell_covered"]:
        return 1.0
    return -0.05
    