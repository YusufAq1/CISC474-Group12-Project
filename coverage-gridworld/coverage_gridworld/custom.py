import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

OBS_MODE = "B"
REWARD_MODE = "_reward_fn4"
GRID_SIZE = 10
_prev_agent_pos = None

_COLORS = np.array([
    [0,   0,   0  ],  # 0 = unexplored (BLACK)
    [255, 255, 255],  # 1 = explored (WHITE)
    [101, 67,  33 ],  # 2 = wall (BROWN)
    [160, 161, 161],  # 3 = agent (GREY)
    [31,  198, 0  ],  # 4 = enemy (GREEN)
    [255, 0,   0  ],  # 5 = unexplored under surveillance (RED)
    [255, 127, 127],  # 6 = explored under surveillance (LIGHT_RED)
], dtype=np.uint8)

# Named aliases for semantic6 readability.
BLACK, WHITE, BROWN, GREY, GREEN, RED, LIGHT_RED = _COLORS


def _mask(grid: np.ndarray, color: np.ndarray) -> np.ndarray:
    """
    Build a binary mask where cells matching the color are 1.0.
    """
    return np.all(grid == color, axis=2).astype(np.float32)


def _encode_grid(grid: np.ndarray) -> np.ndarray:
    flat = grid.reshape(100, 3)
    encoded = np.zeros(100, dtype=np.int32)
    for color_id, color in enumerate(_COLORS):
        encoded[np.all(flat == color, axis=1)] = color_id
    return encoded


def _observation_compact105(grid: np.ndarray) -> np.ndarray:
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

    if OBS_MODE == "A":
        return _obs_space_a()
    
    if OBS_MODE == "B":
        return _obs_space_b()

    raise ValueError(f"Unsupported OBS_MODE: {OBS_MODE}")

# ----- NEW OBS YUSUF ----- #
def _grid_to_ids(grid: np.ndarray) -> np.ndarray:
    
    flat = grid.reshape(100, 3)
    result = np.zeros(100, dtype=np.int32)
    for cid, color in enumerate(_COLORS):
        result[np.all(flat == color, axis=1)] = cid
    return result.reshape(10, 10)


def _obs_a(grid: np.ndarray) -> np.ndarray:
    ids = _grid_to_ids(grid)  # (10, 10)
    channels = np.zeros((6, 10, 10), dtype=np.float32)
    channels[0] = (ids == 0).astype(np.float32)              # unexplored
    channels[1] = np.isin(ids, [1, 3]).astype(np.float32)    # explored
    channels[2] = (ids == 2).astype(np.float32)              # wall
    channels[3] = (ids == 3).astype(np.float32)              # agent position
    channels[4] = (ids == 4).astype(np.float32)              # enemy
    channels[5] = np.isin(ids, [5, 6]).astype(np.float32)    # danger


    agent_cells = np.argwhere(ids == 3)
    unexplored_cells = np.argwhere(ids == 0)
    if len(agent_cells) > 0 and len(unexplored_cells) > 0:
        ar, ac = agent_cells[0]
        dists = np.abs(unexplored_cells[:, 0] - ar) + np.abs(unexplored_cells[:, 1] - ac)
        nearest = unexplored_cells[np.argmin(dists)]
        direction = np.array([(nearest[0] - ar) / 9.0, (nearest[1] - ac) / 9.0],
                             dtype=np.float32)
    else:
        direction = np.zeros(2, dtype=np.float32)

    return np.concatenate([channels.flatten(), direction])  # (602,)


#Davids Obs with prev position global
def _obs_b(grid: np.ndarray) -> np.ndarray:
    global _prev_agent_pos

    ids = _grid_to_ids(grid)

    channels = np.zeros((6, 10, 10), dtype=np.float32)
    channels[0] = (ids == 0).astype(np.float32)
    channels[1] = np.isin(ids, [1, 3]).astype(np.float32)
    channels[2] = (ids == 2).astype(np.float32)
    channels[3] = (ids == 3).astype(np.float32)
    channels[4] = (ids == 4).astype(np.float32)
    channels[5] = np.isin(ids, [5, 6]).astype(np.float32)

    agent_cells = np.argwhere(ids == 3)

    direction = np.zeros(2, dtype=np.float32)
    safe_direction = np.zeros(2, dtype=np.float32)
    danger_density = 0.0
    escape_routes = 0.0
    revisit_flag = 0.0

    if len(agent_cells) > 0:
        ar, ac = agent_cells[0]
        current_pos = ar * 10 + ac

        # --- revisit detection ---
        if _prev_agent_pos is not None and _prev_agent_pos == current_pos:
            revisit_flag = 1.0

        _prev_agent_pos = current_pos  # update AFTER check

        unexplored = np.argwhere(ids == 0)
        safe_unexplored = np.argwhere((ids == 0) & (~np.isin(ids, [5])))

        if len(unexplored) > 0:
            dists = np.abs(unexplored[:, 0] - ar) + np.abs(unexplored[:, 1] - ac)
            nearest = unexplored[np.argmin(dists)]
            direction = np.array([(nearest[0] - ar) / 9.0,
                                  (nearest[1] - ac) / 9.0], dtype=np.float32)

        if len(safe_unexplored) > 0:
            dists = np.abs(safe_unexplored[:, 0] - ar) + np.abs(safe_unexplored[:, 1] - ac)
            nearest = safe_unexplored[np.argmin(dists)]
            safe_direction = np.array([(nearest[0] - ar) / 9.0,
                                       (nearest[1] - ac) / 9.0], dtype=np.float32)

        local = ids[max(0, ar-1):min(10, ar+2), max(0, ac-1):min(10, ac+2)]
        danger_density = np.mean(np.isin(local, [5, 6]).astype(np.float32))

        free = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = ar+dr, ac+dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                if ids[nr, nc] not in [2, 4]:
                    free += 1
        escape_routes = free / 4.0

    return np.concatenate([
        channels.flatten(),
        direction,
        safe_direction,
        [danger_density, escape_routes, revisit_flag]
    ]).astype(np.float32)

def _obs_space_a() -> gym.spaces.Space:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(602,), dtype=np.float32)

def _obs_space_b() -> gym.spaces.Space:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(607,), dtype=np.float32)

def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    global _prev_agent_pos

    if OBS_MODE == "default":
        return grid.flatten()

    if OBS_MODE == "semantic6":
        return _observation_semantic6(grid)

    if OBS_MODE == "compact105":
        return _observation_compact105(grid)

    if OBS_MODE == "A":
        return _obs_a(grid)
    
    if OBS_MODE == "B":
        ids = _grid_to_ids(grid)
        agent_idxs = np.where(ids.flatten() == 3)[0]

        if len(agent_idxs) > 0 and agent_idxs[0] == 0:
            _prev_agent_pos = None

        return _obs_b(grid)

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



def _reward_stealth_safe(info: dict) -> float:
    """
    Reward 3: Stealth + Safe Exploration (Arian)
    Focuses more on avoiding danger than speed
    """

    # Strong penalty if caught
    if info["game_over"]:
        return -90.0

    # Reward for finishing
    if info["cells_remaining"] == 0:
        efficiency = info["steps_remaining"] / 500.0
        return 110.0 + efficiency * 40.0

    # Base reward
    r = 1.2 if info["new_cell_covered"] else -0.1

    # Strong danger avoidance
    agent_pos = info["agent_pos"]
    all_fov_flat = set(
        cell[1] * 10 + cell[0]
        for enemy in info["enemies"]
        for cell in enemy.get_fov_cells()
    )

    adjacent_flat = {agent_pos - 10, agent_pos + 10, agent_pos - 1, agent_pos + 1}
    adjacent_danger = len(adjacent_flat & all_fov_flat)

    # MUCH stronger penalty than other rewards
    r -= adjacent_danger * 0.7

    return r




# ----- NEW REWARD FUNCTION BY YUSUF ------- # 

def _reward_fn3(info: dict) -> float:

    if info["game_over"]:
        return -250.0

    if info["cells_remaining"] == 0:
        
        return 500.0 + info["steps_remaining"] * 0.3

    coverage = info["total_covered_cells"] / info["coverable_cells"]

    if info["new_cell_covered"]:
        
        r = 10.0 + coverage * 15.0
    else:
        
        r = -1.0

    agent_row = info["agent_pos"] // 10
    agent_col = info["agent_pos"] % 10
    fov_cells = set()
    for enemy in info["enemies"]:
        for cell in enemy.get_fov_cells():
            fov_cells.add(cell)

    adjacent_danger = sum(
        1 for (nr, nc) in [
            (agent_row - 1, agent_col),
            (agent_row + 1, agent_col),
            (agent_row,     agent_col - 1),
            (agent_row,     agent_col + 1),
        ]
        if (nr, nc) in fov_cells
    )
    r -= adjacent_danger * 20.0

    r -= 0.2

    return r


#David Reward Implementation
def _reward_fn4(info: dict) -> float:

    if info["game_over"]:
        return -200.0

    if info["cells_remaining"] == 0:
        return 400.0 + info["steps_remaining"] * 0.5
    r = 0.0
    if info["new_cell_covered"]:
        r += 8.0
    else:
        r -= 0.5
    r -= 0.1

    coverage = info["total_covered_cells"] / info["coverable_cells"]
    r += coverage * 5.0
    agent_pos = info["agent_pos"]
    agent_row = agent_pos // 10
    agent_col = agent_pos % 10

    fov_cells = set()
    for enemy in info["enemies"]:
        for cell in enemy.get_fov_cells():
            fov_cells.add(cell)

    adjacent_danger = sum(
        1 for (nr, nc) in [
            (agent_row - 1, agent_col),
            (agent_row + 1, agent_col),
            (agent_row, agent_col - 1),
            (agent_row, agent_col + 1),
        ]
        if (nr, nc) in fov_cells
    )

    r -= adjacent_danger * 8.0
    if (agent_row, agent_col) in fov_cells:
        r -= 15.0

    return r




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
    if REWARD_MODE == "stealth_safe":
        return float(_reward_stealth_safe(info))
    if REWARD_MODE == "_reward_fn3":
        return float(_reward_fn3(info))
    if REWARD_MODE == "_reward_fn4":
        return float(_reward_fn4(info))

    raise ValueError("Unsupported REWARD_MODE: "
                     f"{REWARD_MODE}. Available modes: balanced, main_risk, stealth_safe, _reward_fn3, _reward_fn4")
