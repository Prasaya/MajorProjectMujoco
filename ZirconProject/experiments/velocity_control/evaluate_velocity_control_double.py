"""
Script used for visualizing a policy for the RL transfer velocity control task.
"""
import os.path as osp
import sys
import numpy as np
import torch
from absl import app, flags, logging
from ml_collections.config_flags import config_flags
from gym.spaces import dict as gym_dict

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
task_file = "ZirconProject/experiments/velocity_control/config.py"
config_flags.DEFINE_config_file(
    "task", f"{task_file}:velocity_control_double", "Task")
flags.DEFINE_integer("episode_steps", 1200,
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
    points_to_visit = [
        [4.856604200459673, 1.1405764800292657, 0.0] ,
        [8.330810536821422, 1.8175789650061978, 0.0] ,
        [11.265259091352707, 2.3142911201247567, 0.0] ,
        [14.833479475612702, 2.720844434022592, 0.0] ,
        [17.58882259545875, 1.9530253318653046, 0.0] ,
        [20.293983131983065, -0.6628960579538852, 0.0] ,
        [22.545650991227305, -2.1487207751542785, 0.0] ,
        [25.79037536781569, -2.5086439627976675, 0.0] ,
        [28.279375961465085, -0.6178429732867148, 0.0] ,
        [29.82752085242408, 1.8627263990152652, 0.0] ,
        [31.32733516025359, 3.353534272222852, 0.0] ,
        [34.2201491035111, 3.4439465082597955, 0.0] ,
        [35.69444474157224, 1.366202201932316, 0.0] ,
        [38.55999973929738, 2.4944136284130582, 0.0] ,
        [41.22667493127592, 3.8025266939863926, 0.0] ,
        [44.10138408694276, 1.863421995347591, 0.0] ,
        [46.981951805052155, 1.3228363176487479, 0.0] ,
        [49.67282693662481, -0.16251635455208113, 0.0] ,
        [56.072015626404685, -0.27264860552615744, 0.0] ,
    ]

    env_ctor = dm_control_wrapper.DmControlWrapper.make_env_constructor(
        FLAGS.task.constructor)
    task_kwargs = dict(
        physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=CONTROL_TIMESTEP,
        obstacles=Obstacles(),
        points_to_visit=points_to_visit,
        **FLAGS.task.config
    )
    environment_kwargs = dict(
        time_limit=CONTROL_TIMESTEP*FLAGS.episode_steps,
        random_state=FLAGS.seed
    )
    arena_size = (100., 100.) if FLAGS.big_arena else (8., 8.)
    env = env_ctor(
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        arena_size=arena_size,
        use_walls=True,
        act_noise=0.0,
    )
    # print(env)

    # print(type(env.observation_space))
    # print(env.observation_space.keys())

    obs = env.observation_space
    obs1 = gym_dict.Dict()
    obs2 = gym_dict.Dict()
    for k, v in env.observation_space.items():
        if not k.startswith('walker_1/'):
            obs1[k] = v
    for k, v in env.observation_space.items():
        if not k.startswith('walker/'):
            obs2[k] = v

    print("\n\n\nobs1  ", obs1)
    print("\n\n\nobs2  ", obs2, "\n\n")


    print("Obs1 keys ", list(obs1.keys()), "\n\n")

    # Set up model
    high_level_model = utils.load_policy(
        FLAGS.model_root,
        list(obs1.keys()),
        device=FLAGS.device
    )

   
    # sys.exit()

    high_level_model2 = utils.load_policy(
        FLAGS.model_root,
        list(obs1.keys()),
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
        
        obs1 = {}
        obs2 = {}
        for k, v in obs.items():
            if not k.startswith('walker_1/'):
                obs1[k] = v
        for k, v in obs.items():
            if not k.startswith('walker/'):
                obs2[k] = v

        # print("Type of obs is ", type(obs))
        # print("high level", type(high_level_model.observation_space))
        obs11 = obs1
        if low_level_policy:
            embed, _ = high_level_model.predict(obs1, deterministic=True)
            embed = np.clip(embed, -FLAGS.max_embed, FLAGS.max_embed)
            obs1 = {k: v.astype(np.float32) for k, v in obs1.items()}
            obs1 = obs_as_tensor(obs1, 'cpu')
            embed = torch.tensor(embed)
            action = low_level_policy(obs1, embed)
            action = np.clip(action, -1., 1.)

            # embed2, _ = high_level_model2.predict(obs1, deterministic=True)
            # embed2 = np.clip(embed2, -FLAGS.max_embed, FLAGS.max_embed)
            # obs = {k: v.astype(np.float32) for k, v in obs.items()}
            # obs = obs_as_tensor(obs, 'cpu')
            # embed = torch.tensor(embed)
            # action2 = low_level_policy(obs, embed2)
            # action2 = np.clip(action2, -1., 1.)
        else:
            action, _ = high_level_model.predict(obs1, deterministic=True)
            # action2, _ = high_level_model2.predict(obs2, deterministic=True)
        action2 = action
        return [action, action2]

    if FLAGS.visualize:
        viewer_app = application.Application(
            title='Humanoid Task', width=1024, height=768)
        viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn)


if __name__ == '__main__':
    app.run(main)
