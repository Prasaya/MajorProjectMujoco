from dm_control import composer
# from dm_control.locomotion.examples import basic_cmu_2019
import dm_control.locomotion.tasks.go_to_target 
import numpy as np

import src.examples.basic_cmu_2019 as basic_cmu_2019
import src.examples.cmu_2020_tracking as cmu_2020_tracking

# Build an example environment.
# env = basic_cmu_2019.cmu_humanoid_run_walls()
# env = basic_cmu_2019.cmu_humanoid_maze_forage()
env = cmu_2020_tracking.cmu_humanoid_tracking()
# env = basic_cmu_2019.cmu_humanoid_heterogeneous_forage()

# Get the `action_spec` describing the control inputs.
action_spec = env.action_spec()

# Step through the environment for one episode with random actions.
time_step = env.reset()
while not time_step.last():
  action = np.random.uniform(action_spec.minimum, action_spec.maximum,
                             size=action_spec.shape)
  time_step = env.step(action)
  print("reward = {}, discount = {}, observations = {}.".format(
      time_step.reward, time_step.discount, time_step.observation))


from dm_control import viewer

# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

viewer.launch(environment_loader=env, policy=random_policy)