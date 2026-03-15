import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

_COLORS = np.array([
    [0,   0,   0  ],  # 0 = unexplored (BLACK)
    [255, 255, 255],  # 1 = explored (WHITE)
    [101, 67,  33 ],  # 2 = wall (BROWN)
    [160, 161, 161],  # 3 = agent (GREY)
    [31,  198, 0  ],  # 4 = enemy (GREEN)
    [255, 0,   0  ],  # 5 = unexplored under surveillance (RED)
    [255, 127, 127],  # 6 = explored under surveillance (LIGHT_RED)
], dtype=np.uint8)

def _encode_grid(grid: np.ndarray) -> np.ndarray:

    flat = grid.reshape(100, 3)
    result = np.zeros(100, dtype=np.int32)
    for color_id, color in enumerate(_COLORS):
        result[np.all(flat == color, axis=1)] = color_id
    return result

def observation_space(env: gym.Env) -> gym.spaces.Space:
    # 100 grid + agent_row + agent_col + coverage_frac + adjacent_danger + global_danger
    return gym.spaces.Box(low=0.0, high=1.0, shape=(105,), dtype=np.float32)


def observation(grid: np.ndarray):
    encoded = _encode_grid(grid)                          # shape (100,), int 0-6
    normalized = encoded.astype(np.float32) / 6.0        # normalize to [0, 1]

    # Extract agent position (encoded value 3)
    agent_idxs = np.where(encoded == 3)[0]
    if len(agent_idxs) > 0:
        idx = int(agent_idxs[0])
        agent_row = (idx // 10) / 9.0
        agent_col = (idx % 10) / 9.0
        r, c = idx // 10, idx % 10
        # Local danger: fraction of 4-adjacent cells that are surveillance zones (5 or 6)
        adjacent_vals = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                adjacent_vals.append(encoded[nr * 10 + nc])
        adjacent_danger = sum(1 for v in adjacent_vals if v in (5, 6)) / max(len(adjacent_vals), 1)
    else:
        agent_row, agent_col, adjacent_danger = 0.0, 0.0, 0.0

    # Coverage fraction
    walls = np.sum(encoded == 2)
    total_coverable = max(100 - walls, 1)
    explored = np.sum(encoded == 1) + np.sum(encoded == 3) + np.sum(encoded == 6)
    coverage_frac = float(explored) / total_coverable

    # Global danger: fraction of coverable cells under surveillance
    surveillance = int(np.sum((encoded == 5) | (encoded == 6)))
    global_danger = surveillance / total_coverable

    return np.concatenate([
        normalized,
        [agent_row, agent_col, coverage_frac, adjacent_danger, global_danger]
    ]).astype(np.float32)


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

    # Soft danger penalty: penalize being adjacent to enemy FOV cells
    agent_pos = info["agent_pos"]
    all_fov_flat = set(
        cell[1] * 10 + cell[0]
        for enemy in info["enemies"]
        for cell in enemy.get_fov_cells()
    )
    adjacent_flat = {agent_pos - 10, agent_pos + 10, agent_pos - 1, agent_pos + 1}
    adjacent_danger = len(adjacent_flat & all_fov_flat)

    r = 1.0 if info["new_cell_covered"] else -0.05
    r -= adjacent_danger * 0.2
    return r
    