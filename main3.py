import random
import time
import gymnasium
import shutil
import argparse
import coverage_gridworld  # must be imported, even though it's not directly referenced
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


def human_player():
    # Write the letter for the desired movement in the terminal/console and then press Enter

    input_action = input()
    if input_action.lower() == "w":
        return 3
    elif input_action.lower() == "a":
        return 0
    elif input_action.lower() == "s":
        return 1
    elif input_action.lower() == "d":
        return 2
    elif input_action.isdigit():
        return int(input_action)
    else:
        return 4


def random_player():
    return random.randint(0, 4)


maps = [
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    ],
    [
        [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
    ]
]

# ---- TRAIN ----- #

def train(name, resume=False):

    # create training environment
    train_env = make_vec_env(
        "just_go",
        n_envs=4,
        env_kwargs={
            "predefined_map_list": None
        }
    )

    # create evaluation environment
    eval_env = make_vec_env(
        "just_go",
        n_envs=1,
        env_kwargs={"render_mode": None}
    )

    # if resuming training, load the model
    if resume:
        model = PPO.load(f"./models/{name}/final", env=train_env)
        model.tensorboard_log = f"./logs/{name}"
        print(f"Resuming Training")
    # create PPO agent
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=f"./logs/{name}",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    # callback every eval_freq timesteps
    # evluates the agent for 5 episodes and saves the best model found so far
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{name}/",
        log_path=f"./logs/{name}",
        eval_freq=100_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # the training loop
    model.learn(total_timesteps=1_000_000, callback=eval_callback)
    # save final model
    model.save(f"./models/{name}/final")

    print("Training Complete")

    train_env.close()
    eval_env.close()


# ---- CURRICULUM TRAIN ----- #

# (stage_name, new_map)  — each stage adds one new map to the training pool
_STAGES = [
    ("stage1_just_go",        maps[0]),
    ("stage2_safe",           maps[1]),
    ("stage3_maze",           maps[2]),
    ("stage4_chokepoint",     maps[3]),
    ("stage5_sneaky_enemies", maps[4]),
]


def _train_stage(map_pool, eval_map, run_name, load_from=None):
    """Train on all maps in map_pool (cycling) for 1M steps.

    map_pool  : list of map grids to cycle through during training
    eval_map  : single map grid used for the EvalCallback (the newest/hardest map)
    run_name  : path prefix for logs and checkpoints
    load_from : path to a prior model to load weights from (None = fresh model)
    """
    train_env = make_vec_env(
        "standard",
        n_envs=4,
        env_kwargs={"predefined_map_list": map_pool},
    )
    eval_env = make_vec_env(
        "standard",
        n_envs=1,
        env_kwargs={"predefined_map_list": [eval_map]},
    )

    ppo_kwargs = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=f"./logs/{run_name}",
    )

    if load_from is not None:
        print(f"[{run_name}] Loading model from {load_from}")
        model = PPO.load(load_from, env=train_env)
        model.verbose = 1
        model.tensorboard_log = f"./logs/{run_name}"
    else:
        model = PPO("MlpPolicy", train_env, **ppo_kwargs)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{run_name}/",
        log_path=f"./logs/{run_name}",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    print(f"[{run_name}] Training on {len(map_pool)} map(s) for 1,000,000 steps ...")
    model.learn(total_timesteps=1_000_000, callback=eval_callback, reset_num_timesteps=True)
    model.save(f"./models/{run_name}/final")
    print(f"[{run_name}] Done.")

    train_env.close()
    eval_env.close()

    return f"./models/{run_name}/best_model"


def train_best(name="best_agent", start_stage=0):
    """
    Accumulating curriculum: each stage adds one new map to the training pool.
    The agent always trains on ALL maps seen so far, preventing catastrophic forgetting.

    Stages (0-indexed for --start-stage):
      0  just_go
      1  just_go + safe
      2  just_go + safe + maze
      3  just_go + safe + maze + chokepoint
      4  just_go + safe + maze + chokepoint + sneaky_enemies
    """
    prev_model_path = None

    # If resuming mid-curriculum, derive the load path from the completed stage
    if start_stage > 0:
        prev_stage_name = _STAGES[start_stage - 1][0]
        prev_model_path = f"./models/{name}/{prev_stage_name}/best_model"
        print(f"Resuming from stage {start_stage}, loading: {prev_model_path}")

    accumulated_maps = []
    for i, (stage_name, new_map) in enumerate(_STAGES):
        accumulated_maps.append(new_map)
        run_name = f"{name}/{stage_name}"
        if i < start_stage:
            print(f"Skipping {stage_name} (already completed)")
            prev_model_path = f"./models/{run_name}/best_model"
            continue
        # eval on the newest (hardest) map added this stage
        prev_model_path = _train_stage(
            map_pool=list(accumulated_maps),
            eval_map=new_map,
            run_name=run_name,
            load_from=prev_model_path,
        )

    print("Curriculum training complete!")
    print(f"Final model: ./models/{name}/stage5_sneaky_enemies/best_model")


# ----- Evaluation ------- #

def evaluate(name):

    # load the best model found during training
    try:
        model = PPO.load(f"./models/{name}/best_model")
    except FileNotFoundError:
        model = PPO.load(name)

    # Test agent on the following 3 envs
    test_envs = [
        ("just_go", {"render_mode": "human"}),
        ("safe", {"render_mode": "human"}),
        ("sneaky_enemies", {"render_mode": "human"}),
    ]

    # loop through each test env
    for env_id, kwargs in test_envs:
        print(f"-{env_id}-")
        env = gymnasium.make(env_id, **kwargs)
        # run 3 episodes per environment
        for i in range(1):
            # reset the env and get the initial observation
            obs, info = env.reset()
            done = False
            steps = 0
            # at each step get the action and take a step
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                done = done or truncated
            # compute map coverage and print episode outcome
            coverage = info["total_covered_cells"] / info["coverable_cells"] * 100
            outcome = "CAUGHT" if info["game_over"] else ("DONE" if info["cells_remaining"] == 0 else "TIMEOUT")
            print(f"  Episode {i+1}: {outcome} | Coverage: {coverage:.1f}% | Steps: {steps}")
            time.sleep(1)
        env.close()


# -------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "train_best", "eval"], help="train, train_best, or eval")
    parser.add_argument("--name", default="experiment", help="name for this run (used for saving/loading)")
    parser.add_argument("--resume", action="store_true", help="resume training from the last checkpoint")
    parser.add_argument("--start-stage", type=int, default=0,
                        help="stage index to resume train_best from (0=just_go, 1=safe, 2=maze, 3=chokepoint, 4=sneaky_enemies)")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.name, resume=args.resume)
    elif args.mode == "train_best":
        train_best(name=args.name if args.name != "experiment" else "best_agent",
                   start_stage=args.start_stage)
    else:
        evaluate(args.name)
