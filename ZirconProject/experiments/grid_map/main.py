import os
from absl import app, flags, logging
from ml_collections.config_flags import config_flags
import numpy as np
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm
import pygame
from pygame.locals import *

from ZirconProject.experiments.utils import CheckOccupied
from learn_dqn import MazeEnv

FLAGS = flags.FLAGS
flags.DEFINE_string("model_root", "transfer/velocity_control/locomotion_low_level",
                    "Directory where policy is stored")
task_file = "ZirconProject/experiments/grid_map/config.py"
config_flags.DEFINE_config_file(
    "task", f"{task_file}:velocity_control", "Task")

flags.DEFINE_string(
    "root_dir", "ZirconProject/experiments/grid_map/output", "Path to store all files")
flags.DEFINE_integer("maze_granularity", 2, "Name of the model")
flags.DEFINE_bool("train", False, "Train the model")
flags.DEFINE_integer("num_timesteps", 100000,
                     "Number of timesteps to train for")


logging.set_verbosity(logging.DEBUG)

CONTROL_TIMESTEP = 0.03


def configure_env():
    from dm_control.locomotion.tasks.reference_pose import tracking
    from envs import dm_control_wrapper
    env_ctor = dm_control_wrapper.DmControlWrapper.make_env_constructor(
        FLAGS.task.constructor)
    task_kwargs = dict(
        physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=CONTROL_TIMESTEP,
        obstacles=None,
        points_to_visit=[[5, 5, 0]],
        **FLAGS.task.config
    )
    episode_steps = 100
    environment_kwargs = dict(
        time_limit=CONTROL_TIMESTEP*episode_steps,
        random_state=0
    )
    arena_size = (100., 50.)
    env = env_ctor(
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        arena_size=arena_size,
        use_walls=True,
        act_noise=0.0,
    )
    return env


def configure_maze(env, granularity):
    arena_size = np.array(env._env._task._arena.size) * granularity
    arena_size = arena_size.astype(np.int32)
    x_halfsize = arena_size[0]
    y_halfsize = arena_size[1]
    arena_size *= 2
    maze = np.zeros(arena_size + 1, dtype=np.int8)
    allowed_geoms = ['groundplane', 'walker/root_geom', None]
    check = CheckOccupied(env)
    for i in tqdm(range(-x_halfsize, x_halfsize+1)):
        for j in range(-y_halfsize, y_halfsize+1):
            geom, distance = check([i/granularity, j/granularity, 0])
            if geom not in allowed_geoms:
                # Adjust for pygame coordinate system (0, 0) at top left
                # and mujoco coordinated system (0, 0) at center
                maze[i + x_halfsize, abs(j - y_halfsize)] = 1
    return maze


def main(_):
    os.makedirs(FLAGS.root_dir, exist_ok=True)

    maze = None
    granularity = FLAGS.maze_granularity
    maze_save_path = os.path.join(FLAGS.root_dir, f"maze_{granularity}.npy")
    if os.path.exists(maze_save_path):
        maze = np.load(maze_save_path)
        logging.info("Loaded maze from file")
    else:
        dm_control_env = configure_env()
        maze = configure_maze(dm_control_env, granularity)
        np.save(maze_save_path, maze)
    logging.info(f"Maze shape: {maze.shape}")

    env = MazeEnv(maze)
    # env = Monitor(env)
    eval_env = MazeEnv(maze.copy())
    # eval_env = Monitor(eval_env)
    print('Checking environment:', check_env(env))
    print(A2C.__name__)

    training_algo = A2C
    model = training_algo(
        "MultiInputPolicy",
        env,
        tensorboard_log=FLAGS.root_dir,
        verbose=1)
    model_path = os.path.join(FLAGS.root_dir, "a2c_maze.zip")
    if os.path.exists(model_path):
        model = training_algo.load(model_path, env)
        print("Loaded pretrained model", model_path)

    if FLAGS.train:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(FLAGS.root_dir, "best_model"),
            log_path=FLAGS.root_dir,
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        model.learn(total_timesteps=10_000,
                    log_interval=4,
                    # callback=eval_callback,
                    tb_log_name="DQN",
                    progress_bar=True
                    )
        model.save(model_path)

    obs = env.reset()
    while True:
        quit = False
        for event in pygame.event.get():
            if event.type == QUIT:
                quit = True
        if quit:
            break
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()
          break


if __name__ == "__main__":
    app.run(main)
