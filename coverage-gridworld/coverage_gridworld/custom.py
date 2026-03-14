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

    return gym.spaces.MultiDiscrete(np.full(100, 7, dtype=np.int32))


def observation(grid: np.ndarray):
    
    return _encode_grid(grid)


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
    r = -0.1  
    if info["game_over"]:
        r -= 100
    elif info["cells_remaining"] == 0:
        r += 200
    elif info["new_cell_covered"]:
        r += 10
    else:
        r -= 0.3  
    return r
    