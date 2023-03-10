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
from dm_control.viewer import user_input

from obstacles import Obstacles

FLAGS = flags.FLAGS
flags.DEFINE_string("model_root", "transfer/velocity_control/locomotion_low_level",
                    "Directory where policy is stored")
flags.DEFINE_string("model_root2", "transfer/go_to_target/locomotion_low_level",
                    "Directory where second policy is stored")
flags.DEFINE_float("max_embed", 3., "Maximum embed")
task_file = "ZirconProject/experiments/velocity_control/config.py"
config_flags.DEFINE_config_file(
    "task", f"{task_file}:velocity_control_double", "Task")
flags.DEFINE_integer("episode_steps", 2400,
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
        [8.171408380493276, 0.260292131847196, 0.0] ,
        [13.073279037563736, 2.2173992666894087, 0.0] ,
        [17.60485264382106, 0.9604226807527478, 0.0] ,
        [23.344979593187347, -2.563891498307111, 0.0] ,
        [27.23009416750082, -0.21559164453288027, 0.0] ,
        [31.447607835778726, 1.3813783290414374, 0.0] ,
        [34.95187284029304, -0.41241414860262005, 0.0] ,
        [41.13538038075265, -2.8908069121589968, 0.0] ,
        [44.39022615706183, -3.098403192303951, 0.0] ,
        [50.35637880079873, -1.3894508153701457, 0.0] ,
        [56.958782949701806, 0.1742369209600172, 0.0] ,
        [64.78691213883181, 0.11807382841550762, 0.0] ,
        [77.343361128314, -0.18157823794885797, 0.0] ,
        [80.22523664262047, 0.5315906586163885, 0.0] 
    ]
    points_to_visit2 = [
        [3.9096696394328383, -24.511746674387666, 0.0] ,
        [7.76072685264824, -28.932733716584487, 0.0] ,
        [11.442470599518707, -31.328315751680467, 0.0] ,
        [15.959993960621455, -32.500463284293744, 0.0] ,
        [21.57272275827625, -33.040220192107455, 0.0] ,
        [30.637935895157355, -32.92951738237058, 0.0] ,
        [39.22340598017345, -32.4134799116287, 0.0] ,
        [51.29876068433137, -31.241269543014297, 0.0] ,
        [59.943503482964886, -30.721026478942576, 0.0] ,
    ]
    points_to_visit2 = [
        [8., -20., 0],
        [12.934117473957759, -17.43890235591438, 0.0] ,
        [16.346919582415097, -17.429689260557378, 0.0] ,
        [23.4912619626964, -23.03255252258321, 0.0] ,
        [26.272894364298573, -22.974454853440847, 0.0] ,
        [30.910419663645524, -18.69125932569595, 0.0] ,
        [34.52111266594255, -18.735201423463575, 0.0] ,
        [41.52674422990249, -23.192955051340228, 0.0] ,
        [46.748970891445474, -23.140558508243313, 0.0] ,
        [54.47574930198334, -21.554818208896002, 0.0] ,
        [62.981489588913526, -21.049342059939164, 0.0] ,
    ]

    env_ctor = dm_control_wrapper.DmControlWrapper.make_env_constructor(
        FLAGS.task.constructor)
    task_kwargs = dict(
        physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=CONTROL_TIMESTEP,
        obstacles=Obstacles(),
        points_to_visit=points_to_visit,
        points_to_visit2=points_to_visit2,
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

    obs = env.observation_space
    obs1 = gym_dict.Dict()
    obs2 = gym_dict.Dict()

    for k, v in env.observation_space.items():
        if not k.startswith('walker_1/') and k != 'target_obs2':
            obs1[k] = v
    for k, v in env.observation_space.items():
        if k.startswith('target_obs2'):
            obs2[k.replace('target_obs2', 'target_obs')] = v
        elif not k.startswith('walker/'):
            obs2[k.replace('walker_1/', 'walker/')] = v

    # Set up model
    high_level_model = utils.load_policy(
        FLAGS.model_root,
        list(obs1.keys()),
        device=FLAGS.device
    )

    high_level_model2 = utils.load_policy(
        FLAGS.model_root,
        list(obs2.keys()),
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
            if not k.startswith('walker_1/') and k != 'target_obs2':
                obs1[k] = v
        for k, v in obs.items():
            if k == 'target_obs2':
                obs2['target_obs'] = v
            elif not k.startswith('walker/'):
                obs2[k.replace('walker_1/', 'walker/')] = v

        if low_level_policy:
            embed, _ = high_level_model.predict(obs1, deterministic=True)
            embed = np.clip(embed, -FLAGS.max_embed, FLAGS.max_embed)
            obs1 = {k: v.astype(np.float32) for k, v in obs1.items()}
            obs1 = obs_as_tensor(obs1, 'cpu')
            embed = torch.tensor(embed)
            action = low_level_policy(obs1, embed)
            action = np.clip(action, -1., 1.)

            embed2, _ = high_level_model2.predict(obs2, deterministic=True)
            embed2 = np.clip(embed2, -FLAGS.max_embed, FLAGS.max_embed)
            obs2 = {k: v.astype(np.float32) for k, v in obs2.items()}
            obs2 = obs_as_tensor(obs2, 'cpu')
            embed2 = torch.tensor(embed2)
            action2 = low_level_policy(obs2, embed2)
            action2 = np.clip(action2, -1., 1.)
        else:
            action, _ = high_level_model.predict(obs1, deterministic=True)
            action2, _ = high_level_model2.predict(obs2, deterministic=True)
        # action2 = action
        return [action, action2]

    if FLAGS.visualize:
        viewer_app = application.Application(
            title='Humanoid Task', width=1024, height=768)

        def custom_handler():
            task = env._task
            body_id, position = viewer_app._viewer.camera.raycast(
                viewer_app._viewport, viewer_app._viewer._mouse.position)
            if position is not None:
                print(list(position), ',')
            else:
                print(position)
            task.points_to_visit2.append(position)
            # print(env.physics.model.id2name(body_id, 'geom'))

        viewer_app.register_callback(
            custom_handler, (user_input.MOUSE_BUTTON_LEFT, user_input.MOD_SHIFT))

        viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn)


if __name__ == '__main__':
    app.run(main)
