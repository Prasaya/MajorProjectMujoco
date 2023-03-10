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
        [7.611853438163575, -0.5073105305616501, 7.105427357601002e-15] ,
        [12.861446781298447, 2.3271483232732653, -7.105427357601002e-15] ,
        [17.325426497840677, 1.1029807763040917, -7.105427357601002e-15] ,
        [22.963621618651896, -2.3824328845690754, -7.105427357601002e-15] ,
        [27.20037137504983, -0.7906175779270974, 0.0] ,
        [31.09778261406389, 0.9390053147846729, 0.0] ,
        [35.07275869070873, 0.40137054658193705, 7.105427357601002e-15] ,
        [41.92435270948895, -2.667303181056237, 0.0] ,
        [47.74156407708054, -3.0610189835335184, 0.0] ,
        [59.45508018705748, -0.4856270377874772, 0.0] ,
        [65.89125074654154, -0.6719150763542157, 7.105427357601002e-15] ,
    ]
    points_to_visit2 = [
        [8.459533714357882, -20.601235216654704, 0.0] ,
        [12.948384942549517, -18.072087396979345, 0.0] ,
        [16.85135137371646, -18.36018018044958, 0.0] ,
        [22.898667818265924, -22.861297006564726, 0.0] ,
        [27.857167204500513, -21.533089375771045, 0.0] ,
        [31.481044283597583, -19.204959105868376, 0.0] ,
        [36.10544475527765, -19.631789512272466, 0.0] ,
        [42.46201862328934, -23.52524543481887, 0.0] ,
        [53.146555706168286, -22.43094726605406, 0.0] ,
        [64.85964375781808, -20.32794733744174, 0.0] ,
        [68.38786047992997, -19.852334870654296, 0.0] ,
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
        if not k.startswith('walker_1/'):
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
            if not k.startswith('walker_1/'):
                obs1[k] = v
        for k, v in obs.items():
            if k.startswith('target_obs2'):
                obs2[k.replace('target_obs2', 'target_obs')] = v
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
            # task.points_to_visit.append(position)
            # print(env.physics.model.id2name(body_id, 'geom'))

        viewer_app.register_callback(
            custom_handler, (user_input.MOUSE_BUTTON_LEFT, user_input.MOD_SHIFT))

        viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn)


if __name__ == '__main__':
    app.run(main)
