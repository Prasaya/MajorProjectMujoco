"""
Configs for the training script. Uses the ml_collections config library.
"""
from ml_collections import ConfigDict
import velocity_control


def get_config(task_string):
    tasks = {
        'velocity_control': ConfigDict({
            'constructor': velocity_control.VelocityControl,
            'config': ConfigDict(dict(
                max_speed=4.5,
                reward_margin=0.75,
                direction_exponent=1.,
                steps_before_changing_velocity=1
            ))
        })
    }

    return tasks[task_string]
