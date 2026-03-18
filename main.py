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
        ("chockepoint", {"render_mode": 'human'}),
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

def train_competition():
    pass

def evaluate_best():
    pass


# -------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "train_best", "eval"], help="train, train_best, or eval")
    parser.add_argument("--name", default="experiment", help="name for this run (used for saving/loading)")
    parser.add_argument("--resume", action="store_true", help="resume training from the last checkpoint")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.name, resume=args.resume)
    elif args.mode == "train_best":
        train_best(name=args.name if args.name != "experiment" else "best_agent")
    else:
        evaluate(args.name)