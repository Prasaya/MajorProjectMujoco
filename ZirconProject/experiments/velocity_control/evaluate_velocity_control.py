"""
Script used for visualizing a policy for the RL transfer velocity control task.
"""
import os.path as osp
import numpy as np
import torch
from absl import app, flags, logging
from ml_collections.config_flags import config_flags

from ZirconProject.custom_application import application
from dm_control.locomotion.tasks.reference_pose import tracking
from stable_baselines3.common.utils import obs_as_tensor
from mocapact.sb3 import utils
from envs import dm_control_wrapper
from mocapact.distillation import model

from obstacles import Obstacles

FLAGS = flags.FLAGS
flags.DEFINE_string("model_root", "transfer/velocity_control/locomotion_low_level",
                    "Directory where policy is stored")
flags.DEFINE_float("max_embed", 3., "Maximum embed")
task_file = "ZirconProject/experiments/follow_target/config.py"
config_flags.DEFINE_config_file(
    "task", f"{task_file}:velocity_control", "Task")
flags.DEFINE_integer("episode_steps", 833,
                     "Number of time steps in an episode")

# Visualization hyperparameters
flags.DEFINE_bool("visualize", True, "Whether to visualize via GUI")
flags.DEFINE_bool("big_arena", True,
                  "Whether to use a bigger arena for visualization")

# Evaluation hyperparameters
flags.DEFINE_integer("n_eval_episodes", 0,
                     "Number of episodes to numerically evaluate policy")
flags.DEFINE_integer("n_workers", 1, "Number of parallel workers")
flags.DEFINE_bool("always_init_at_clip_start", False,
                  "Whether to initialize at beginning or random point in clip")
flags.DEFINE_float("termination_error_threshold", 0.3,
                   "Error for cutting off expert rollout")
flags.DEFINE_integer("seed", 0, "RNG seed")
flags.DEFINE_string("save_path", None,
                    "If desired, the path to save the evaluation results")
flags.DEFINE_string("device", "auto", "Device to run evaluation on")


logging.set_verbosity(logging.WARNING)

CONTROL_TIMESTEP = 0.03


def main(_):
    env_ctor = dm_control_wrapper.DmControlWrapper.make_env_constructor(
        FLAGS.task.constructor)
    task_kwargs = dict(
        physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=CONTROL_TIMESTEP,
        obstacles=Obstacles(),
        **FLAGS.task.config
    )
    environment_kwargs = dict(
        time_limit=CONTROL_TIMESTEP*FLAGS.episode_steps,
        random_state=FLAGS.seed
    )
    arena_size = (50., 50.) if FLAGS.big_arena else (8., 8.)
    env = env_ctor(
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        arena_size=arena_size,
        use_walls=True,
        act_noise=0.0,
    )
    print(env)

    # Set up model
    high_level_model = utils.load_policy(
        FLAGS.model_root,
        list(env.observation_space.keys()),
        device=FLAGS.device
    )

    if osp.exists(osp.join(FLAGS.model_root, 'low_level_policy.ckpt')):
        distilled_model = model.NpmpPolicy.load_from_checkpoint(
            osp.join(FLAGS.model_root, 'low_level_policy.ckpt'),
            map_location='cpu'
        )
        low_level_policy = distilled_model.low_level_policy
    else:
        low_level_policy = None

    @torch.no_grad()
    def policy_fn(time_step):
        obs = env.get_observation(time_step)
        if low_level_policy:
            embed, _ = high_level_model.predict(obs, deterministic=True)
            embed = np.clip(embed, -FLAGS.max_embed, FLAGS.max_embed)
            obs = {k: v.astype(np.float32) for k, v in obs.items()}
            obs = obs_as_tensor(obs, 'cpu')
            embed = torch.tensor(embed)
            action = low_level_policy(obs, embed)
            action = np.clip(action, -1., 1.)
            # print(action.shape)
        else:
            action, _ = high_level_model.predict(obs, deterministic=True)
        return action

    if FLAGS.visualize:
        viewer_app = application.Application(
            title='Humanoid Task', width=1024, height=768)
        viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn)


if __name__ == '__main__':
    app.run(main)
