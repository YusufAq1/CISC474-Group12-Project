import random
import time
import gymnasium
import shutil
import argparse
import coverage_gridworld  # must be imported, even though it's not directly referenced
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt


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

extra_maps = [
    # Map 5 — open field, 1 central enemy
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 6 — two rooms connected by corridor, 2 enemies
    [
        [3, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 2, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
    ],
    # Map 7 — L-shaped walls, 3 enemies
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0, 4, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 8 — scattered walls, 3 enemies spread out
    [
        [3, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
    ],
    # Map 9 — 4 corner enemies, few walls
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 10 — grid pattern walls, 2 enemies (mini sneaky)
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 11 — horizontal corridors, 3 enemies
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 0, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, 0, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 12 — dense walls, 1 enemy (navigation challenge)
    [
        [3, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 2, 0, 2, 0, 0, 0],
        [2, 2, 0, 0, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 2, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 13 — asymmetric, 3 enemies at edges
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0]
    ],
    # Map 14 — ring of walls, 2 inner enemies
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 4, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 4, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 2, 0, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Map 15 — 5 enemies, open (hardest training map)
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0]
    ],
]


all_maps = maps + extra_maps

# ---- TRAIN ----- #

def train(name, resume=False):
    
    # create training environment
    train_env = make_vec_env(
        "standard",
        n_envs=5,
        env_kwargs={
            "predefined_map_list": maps
        }
    )

    # create evaluation environment
    eval_env = make_vec_env(
        "standard",
        n_envs=5,
        env_kwargs={"render_mode": None, "predefined_map_list": maps}
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
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # the training loop
    model.learn(total_timesteps=500_000, callback=eval_callback)
    # save final model
    model.save(f"./models/{name}/final")

    print("Training Complete")

    train_env.close()
    eval_env.close()

# ----- Evaluation ------- #

def evaluate(name):

    # load the best model found during training
    try:
        model = PPO.load(f"./models/{name}/best_model")
    except FileNotFoundError:
        model = PPO.load(name)

    # Test agent on the following 5 envs 
    test_envs = [
        ("just_go", {"render_mode": 'human'}),
        ("safe", {"render_mode": 'human'}),
        ("maze", {"render_mode": 'human'}),
        ("chokepoint", {"render_mode": 'human'}),
        ("sneaky_enemies", {"render_mode": 'human'}),
    ]

    # ---- ADDED: storage for plots ----
    all_coverages = []
    all_steps = []
    labels = []

    # loop through each test env
    for env_id, kwargs in test_envs:
        print(f"-{env_id}-")
        env = gymnasium.make(env_id, **kwargs)

        # run 1 episodes per environment
        for i in range(1):
            obs, info = env.reset()
            done = False
            steps = 0 

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                done = done or truncated

            # compute map coverage
            coverage = info["total_covered_cells"] / info["coverable_cells"] * 100

            outcome = "CAUGHT" if info["game_over"] else (
                "DONE" if info["cells_remaining"] == 0 else "TIMEOUT"
            )

            print(f"  Episode {i+1}: {outcome} | Coverage: {coverage:.1f}% | Steps: {steps}")

            # ---- ADDED: store results ----
            all_coverages.append(coverage)
            all_steps.append(steps)
            labels.append(f"{env_id}-{i+1}")

            time.sleep(1)

        env.close()


    # Coverage plot
    plt.figure()
    plt.bar(labels, all_coverages)
    plt.xticks(rotation=45)
    plt.ylabel("Coverage (%)")
    plt.title("Coverage per Episode")
    plt.tight_layout()
    plt.savefig("coverage_plot.png")

    # Steps plot
    plt.figure()
    plt.bar(labels, all_steps)
    plt.xticks(rotation=45)
    plt.ylabel("Steps")
    plt.title("Steps per Episode")
    plt.tight_layout()
    plt.savefig("steps_plot.png")

    print("Plots saved: coverage_plot.png, steps_plot.png")
    
# ------ TRAINING THE BEST AGENT FOR COMPETITION ------ #

def train_competition(name="best_agent", resume=False, extra_timesteps=3_000_000):

    eval_env = make_vec_env(
        "standard",
        n_envs=1,
        env_kwargs={"predefined_map_list": maps},
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{name}/",
        log_path=f"./logs/{name}/",
        eval_freq=25_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    if resume:
        
        checkpoint = f"./models/{name}/final"
        resume_maps = all_maps
        print(f"\nResuming from {checkpoint}.zip — {extra_timesteps:,} steps on all maps")

        train_env = make_vec_env(
            "standard",
            n_envs=4,
            env_kwargs={"predefined_map_list": resume_maps},
        )
        model = PPO.load(checkpoint, env=train_env)
        model.tensorboard_log = f"./logs/{name}"
        model.learn(
            total_timesteps=extra_timesteps,
            callback=[eval_callback, cov_callback],
            reset_num_timesteps=False,  
        )
        model.save(f"./models/{name}/final")
        print(f"Resumed training complete: {name}")
        train_env.close()

    else:
        
        stages = [
            (maps[0:2],  800_000),    # Stage 1: just_go + safe  
            (maps[0:3],  800_000),    # Stage 2: + maze           
            (maps[0:4],  1_500_000),  # Stage 3: + chokepoint     
            (all_maps,   3_000_000),  # Stage 4: all 16 maps      
        ]

        model = None

        for stage_idx, (stage_maps, stage_steps) in enumerate(stages):
            print(f"\n{'='*60}")
            print(f"  Stage {stage_idx + 1}/{len(stages)}: "
                  f"{stage_steps:,} steps | {len(stage_maps)} maps")
            print(f"{'='*60}")

            train_env = make_vec_env(
                "standard",
                n_envs=4,
                env_kwargs={"predefined_map_list": stage_maps},
            )

            if model is None:
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    verbose=1,
                    tensorboard_log=f"./logs/{name}",
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=256,
                    n_epochs=10,
                    gamma=0.995,
                    gae_lambda=0.95,
                    ent_coef=0.02,
                    clip_range=0.2,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                )
            else:
                model.set_env(train_env)

            model.learn(
                total_timesteps=stage_steps,
                callback=eval_callback,
                reset_num_timesteps=(stage_idx == 0),
            )

            model.save(f"./models/{name}/stage_{stage_idx + 1}")
            print(f"  Stage {stage_idx + 1} checkpoint saved.")
            train_env.close()

        model.save(f"./models/{name}/final")
        print(f"\nCurriculum training complete: {name}")

    best_src = f"./models/{name}/best_model.zip"
    if os.path.exists(best_src):
        shutil.copy(best_src, f"./{name}.zip")
        print(f"Best model copied to ./{name}.zip")

    eval_env.close()

def evaluate_best(name="best_agent"):
    # load the best model found during training
    try:
        model = PPO.load(f"./models/{name}/best_model")
    except FileNotFoundError:
        model = PPO.load(name)

    # Test agent on the following 5 envs 
    test_envs = [
        ("just_go", {"render_mode": 'human'}),
        ("safe", {"render_mode": 'human'}),
        ("maze", {"render_mode": 'human'}),
        ("chokepoint", {"render_mode": 'human'}),
        ("sneaky_enemies", {"render_mode": 'human'}),
    ]

    # add extra maps to test_envs
    for i, m in enumerate(extra_maps):
        test_envs.append(("standard", {"render_mode": 'human', "predefined_map_list": [m]}))

    # loop through each test env
    for idx, (env_id, kwargs) in enumerate(test_envs):
        label = env_id if env_id != "standard" else f"extra_map_{idx - 4}"
        print(f"-{label}-")
        env = gymnasium.make(env_id, **kwargs)

        # run 1 episodes per environment
        for i in range(1):
            obs, info = env.reset()
            done = False
            steps = 0 

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                done = done or truncated

            # compute map coverage
            coverage = info["total_covered_cells"] / info["coverable_cells"] * 100

            outcome = "CAUGHT" if info["game_over"] else (
                "DONE" if info["cells_remaining"] == 0 else "TIMEOUT"
            )

            print(f"  Episode {i+1}: {outcome} | Coverage: {coverage:.1f}% | Steps: {steps}")


            time.sleep(1)

        env.close()


# -------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "train_competition", "eval", "evaluate_best"], help="train, train_competition, eval, or evaluate_best")
    parser.add_argument("--name", default="experiment", help="name for this run (used for saving/loading)")
    parser.add_argument("--resume", action="store_true", help="resume training from the last checkpoint")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.name, resume=args.resume)
    elif args.mode == "train_competition":
        train_competition(name=args.name, resume=args.resume)
    elif args.mode == "evaluate_best":
        evaluate_best(args.name)
    else:
        evaluate(args.name)
