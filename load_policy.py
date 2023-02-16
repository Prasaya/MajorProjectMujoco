from mocapact import observables
from mocapact.sb3 import utils
import numpy as np
from dm_control.viewer import application

model_name = "CMU_049_08"
start_step = 0
end_step = 156

expert_path = f"/home/prasaya/cs-projects/mocapact/MoCapAct/data/experts/{model_name}-{start_step}-{end_step}/eval_rsi/model"
expert = utils.load_policy(expert_path, observables.TIME_INDEX_OBSERVABLES)

from mocapact.envs import tracking
from dm_control.locomotion.tasks.reference_pose import types
dataset = types.ClipCollection(ids=[model_name], start_steps=[start_step], end_steps=[end_step])
env = tracking.MocapTrackingGymEnv(dataset)
obs, done = env.reset(), False

state = None
def policy_fn(time_step):
    global state
    if time_step.step_type == 0: # first time step
        state = None
    action, state = expert.predict(env.get_observation(time_step), state, deterministic=True)
    return action

# from dm_control import viewer
# viewer.launch(env.dm_env, policy=policy_fn)

viewer_app = application.Application(title='Output', width=1024, height=768)
viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn)




# action_array = []
# while not done:
#     # for i in range(max_frame):
#     env.physics.render()
#     action, _ = expert.predict(obs, deterministic=True)
#     # print("action:", action.shape)
#     obs, rew, done, _ = env.step(action)
#     action_array.append(np.array(action))

#         # video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
#         #                     env.physics.render(height, width, camera_id=1)])
#         # print(obs)
#         # print(rew)
#     # for i in range(max_frame):
#     #     img = plt.imshow(video[i])
#     #     plt.pause(0.01)  # Need min display time > 0.0.
#     #     plt.draw()



# from dm_control import suite
# from dm_control import viewer

# env = suite.load(domain_name="humanoid", task_name="stand")
# action_spec = env.action_spec()
# print("action_spec:", action_spec)

# # Define a uniform random policy.
# def random_policy(time_step):
#     print("time_step " , time_step)
#     del time_step  # Unused.
#     return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)

# print("\n\nrandom something\n", np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape))

# # Launch the viewer application.
# viewer.launch(env, policy=random_policy)
# # viewer.launch(environment_loader=env)




# from dm_control import suite
# import matplotlib.pyplot as plt
# import numpy as np

# max_frame = 90

# width = 480
# height = 480
# video = np.zeros((90, height, 2 * width, 3), dtype=np.uint8)

# # Load one task:
# env = suite.load(domain_name="cartpole", task_name="swingup")

# # Step through an episode and print out reward, discount and observation.
# action_spec = env.action_spec()
# time_step = env.reset()
# while not time_step.last():
#   for i in range(max_frame):
#     action = np.random.uniform(action_spec.minimum,
#                              action_spec.maximum,
#                              size=action_spec.shape)
#     time_step = env.step(action)
#     video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
#                           env.physics.render(height, width, camera_id=1)])
#     #print(time_step.reward, time_step.discount, time_step.observation)
#   for i in range(max_frame):
#     img = plt.imshow(video[i])
#     plt.pause(0.01)  # Need min display time > 0.0.
#     plt.draw()