import numpy as np
from dm_control import mujoco


class CheckOccupied:
    def __init__(self, env):
        self.env = env
        self.physics = env._env.physics
        mujoco.mj_kinematics(self.physics.model.ptr, self.physics.data.ptr)

    def __call__(self, point_to_check):
        ray_from = np.array(point_to_check, dtype=np.float64)
        ray_from[2] = max(ray_from[2], 50.0)
        ray_dir = np.array([0, 0, -1], dtype=np.float64)
        geom_ids = np.array([-1], dtype=np.intc)
        distance = mujoco.wrapper.mjbindings.mjlib.mj_ray(
            self.physics.model.ptr, self.physics.data.ptr, ray_from, ray_dir,
            None, 1, -1, geom_ids)

        geom = None
        id = geom_ids[0]
        if id != -1:
            geom = self.physics.model.id2name(id, 'geom')

        return geom, distance
