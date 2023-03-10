"""
The velocity control task.
"""
import collections
import tree
import os.path as osp
import numpy as np
import mujoco

from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable as dm_observable
from dm_control.locomotion.tasks.reference_pose import tracking, utils
from dm_control.mjcf import Physics
from dm_control.locomotion.walkers import cmu_humanoid, initializers
from dm_control.locomotion.mocap import cmu_mocap_data, loader


class StandInitializer(initializers.WalkerInitializer):
    def __init__(self):
        ref_path = cmu_mocap_data.get_path_for_cmu(version='2020')
        mocap_loader = loader.HDF5TrajectoryLoader(ref_path)
        trajectory = mocap_loader.get_trajectory('CMU_040_12')
        clip_reference_features = trajectory.as_dict()
        clip_reference_features = tracking._strip_reference_prefix(
            clip_reference_features, 'walker/')
        self._stand_features = tree.map_structure(
            lambda x: x[0], clip_reference_features)

    def initialize_pose(self, physics, walker, random_state):
        del random_state
        utils.set_walker_from_features(physics, walker, self._stand_features)
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)


class VelocityControl(composer.Task):
    """
    A task that requires the walker to track a randomly changing velocity.
    """

    def __init__(
        self,
        walker,
        arena,
        max_speed=4.5,
        reward_margin=0.75,
        direction_exponent=1.,
        steps_before_changing_velocity=83,
        physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=0.03,
        obstacles=None,
        points_to_visit=[],
        points_to_visit2=[],
    ):
        self._obstacles = obstacles
        self._walker = walker
        initializer = StandInitializer()
        self._walker2 = cmu_humanoid.CMUHumanoidPositionControlledV2020(
            initializer=initializer)

        self._arena = arena
        self._walker.create_root_joints(self._arena.attach(self._walker))

        self._walker2.create_root_joints(self._arena.attach(self._walker2))

        self._max_speed = max_speed
        self._reward_margin = reward_margin
        self._direction_exponent = direction_exponent
        self._steps_before_changing_velocity = steps_before_changing_velocity
        self._move_speed = 0.
        self._move_angle = 0.
        self._move_speed_counter = 0.

        self._task_observables = collections.OrderedDict()

        def task_state(physics):
            del physics
            sin, cos = np.sin(self._move_angle), np.cos(self._move_angle)
            phase = self._move_speed_counter / self._steps_before_changing_velocity
            return np.array([self._move_speed, sin, cos, phase])
        self._task_observables['target_obs'] = dm_observable.Generic(
            task_state)
        

        self._move_speed2 = 0.
        self._move_angle2 = 0.
        self._move_speed_counter2 = 0.
        self._task_observables2 = collections.OrderedDict()

        def task_state2(physics):
            del physics
            sin2, cos2 = np.sin(self._move_angle2), np.cos(self._move_angle2)
            phase2 = self._move_speed_counter2 / self._steps_before_changing_velocity
            return np.array([self._move_speed, sin2, cos2, phase2])
        self._task_observables2['target_obs2'] = dm_observable.Generic(
            task_state2)


        enabled_observables = []
        enabled_observables += self._walker.observables.proprioception
        enabled_observables += self._walker.observables.kinematic_sensors
        enabled_observables += self._walker.observables.dynamic_sensors
        enabled_observables.append(self._walker.observables.sensors_touch)
        enabled_observables.append(self._walker.observables.torso_xvel)
        enabled_observables.append(self._walker.observables.torso_yvel)
        enabled_observables += list(self._task_observables.values())
        for obs in enabled_observables:
            obs.enabled = True

        enabled_observables2 = []
        enabled_observables2 += self._walker2.observables.proprioception
        enabled_observables2 += self._walker2.observables.kinematic_sensors
        enabled_observables2 += self._walker2.observables.dynamic_sensors
        enabled_observables2.append(self._walker2.observables.sensors_touch)
        enabled_observables2.append(self._walker2.observables.torso_xvel)
        enabled_observables2.append(self._walker2.observables.torso_yvel)
        enabled_observables2 += list(self._task_observables2.values())
        for obs in enabled_observables2:
            obs.enabled = True

        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

        self.points_to_visit = points_to_visit
        self.dir_index = 0
        self._reward_step_counter = 0

        self.points_to_visit2 = points_to_visit2
        self.dir_index2 = 0
        self._reward_step_counter2 = 0

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def _is_disallowed_contact(self, contact):
        set1, set2 = self._walker_nonfoot_geomids, self._ground_geomids
        set3, set4 = self._walker2_nonfoot_geomids, self._ground_geomids
        return (((contact.geom1 in set1 and contact.geom2 in set2) or
                 (contact.geom1 in set2 and contact.geom2 in set1))
                # #Changed here
                and ((contact.geom1 in set3 and contact.geom2 in set4) or
                     (contact.geom1 in set4 and contact.geom2 in set3))
                )

    def _sample_move_speed(self, random_state, physics):
        # Static directions (steps_before_changing_velocity=50 in config.py)
        # source = 0
        # dir = [np.pi*1.5, 0, 0, np.pi*1/6, 0, 0, 0, 0, np.pi*5/3, np.pi*11/6, 0, 0, np.pi*5/12, np.pi*1/3, np.pi*0.105,
        #        np.pi*35/18, np.pi*11/6, np.pi*1/4, np.pi*1/4, np.pi*1/6, np.pi*7/4, 0]
        # if(self.dir_index < len(dir)):
        #     source = dir[self.dir_index]
        #     self.dir_index += 1
        # else:
        #     source = dir[-1]
        # print("Changing source to ", source)

        # Dynamic direction
        agent_pos = physics.named.data.xpos['walker/root']
        agent_pos = np.array([agent_pos[0], agent_pos[1]])
        required_pos = self.points_to_visit[0]
        for pos in self.points_to_visit:
            if pos[0] > agent_pos[0]:
                required_pos = pos
                break
        print("Moving to ", required_pos, "from", agent_pos)
        source = np.arctan2(
            required_pos[1] - agent_pos[1], required_pos[0] - agent_pos[0])
        if source < 0:
            source += 2*np.pi
        if agent_pos[0] < 1:
            source = 1.5*np.pi
        # print("Changing source to", np.rad2deg(source), "for moving to",
        #       required_pos, "from", agent_pos)
        # print("Current move angle: ", np.rad2deg(self._move_angle))

        # self._move_speed = random_state.uniform(high=self._max_speed)
        # self._move_angle = random_state.uniform(high=2*np.pi)
        self._move_speed = 2
        self._move_angle = source
        self._move_speed_counter = 0

    def _sample_move_speed2(self, random_state, physics):
        agent_pos2 = physics.named.data.xpos['walker_1/root']
        agent_pos2 = np.array([agent_pos2[0], agent_pos2[1]])
        required_pos2 = self.points_to_visit2[0]
        for pos2 in self.points_to_visit2:
            if pos2[0] > agent_pos2[0]:
                required_pos2 = pos2
                break
        print("Moving to for agent2 ", required_pos2, "from", agent_pos2, "\n\n")
        source2 = np.arctan2(
            required_pos2[1] - agent_pos2[1], required_pos2[0] - agent_pos2[0])
        if source2 < 0:
            source2 += 2*np.pi
        if agent_pos2[0] < 1:
            source2 = 1.5*np.pi
        
        self._move_speed2 = 2
        self._move_angle2 = source2
        self._move_speed_counter2 = 0

    def should_terminate_episode(self, physics):
        del physics
        return self._failure_termination

    def get_discount(self, physics):
        del physics
        if self._failure_termination:
            return 0.
        else:
            return 1.

    def initialize_episode(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)
        self._sample_move_speed(random_state, physics)
        self._sample_move_speed2(random_state, physics)

        self._failure_termination = False

        walker_foot_geoms = set(self._walker.ground_contact_geoms)
        walker_nonfoot_geoms = [
            geom for geom in self._walker.mjcf_model.find_all('geom')
            if geom not in walker_foot_geoms
        ]
        self._walker_nonfoot_geomids = set(
            physics.bind(walker_nonfoot_geoms).element_id)

        self._walker2.reinitialize_pose(physics, random_state)
        walker2_foot_geoms = set(self._walker2.ground_contact_geoms)
        walker2_nonfoot_geoms = [
            geom for geom in self._walker2.mjcf_model.find_all('geom')
            if geom not in walker2_foot_geoms
        ]
        self._walker2_nonfoot_geomids = set(
            physics.bind(walker2_nonfoot_geoms).element_id)

        self._ground_geomids = set(physics.bind(
            self._arena.ground_geoms).element_id)

        rotation = 0.5 * np.pi
        quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
        walker_x, walker_y = 0, 0

        self._walker.shift_pose(
            physics,
            position=[walker_x, walker_y, 0.],
            quaternion=quat,
            rotate_velocity=True)

        self._walker2.shift_pose(
            physics,
            position=[0., -20., 0.],
            quaternion=quat,
            rotate_velocity=True)

    def get_reward(self, physics):
        xvel = self._walker.observables.torso_xvel(physics)
        yvel = self._walker.observables.torso_yvel(physics)
        speed = np.linalg.norm([xvel, yvel])
        speed_error = self._move_speed - speed
        speed_reward = np.exp(-(speed_error / self._reward_margin)**2)
        if np.isclose(xvel, 0.) and np.isclose(yvel, 0.):
            angle_reward = 1.
        else:
            direction = np.array([xvel, yvel])
            direction /= np.linalg.norm(direction)
            direction_tgt = np.array(
                [np.cos(self._move_angle), np.sin(self._move_angle)])
            dot = direction_tgt.dot(direction)
            angle_reward = ((dot + 1) / 2)**self._direction_exponent

        reward = speed_reward * angle_reward

        return reward

    def before_step(self, physics, action, random_state):
        self._walker.apply_action(physics, action[0], random_state)
        self._walker2.apply_action(physics, action[1], random_state)

    def after_step(self, physics, random_state):
        self._failure_termination = False
        for contact in physics.data.contact:
            if self._is_disallowed_contact(contact):
                self._failure_termination = True
                break

        self._move_speed_counter += 1
        if self._move_speed_counter >= self._steps_before_changing_velocity:
            self._sample_move_speed(random_state, physics)

        self._move_speed_counter2 += 1
        if self._move_speed_counter2 >= self._steps_before_changing_velocity:
            self._sample_move_speed2(random_state, physics)