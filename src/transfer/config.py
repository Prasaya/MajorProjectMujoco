"""
Configs for the training script. Uses the ml_collections config library.
"""
from ml_collections import ConfigDict
from mocapact.tasks import velocity_control
from mocapact.tasks import go_to_target

def get_config(task_string):
    tasks = {
        'go_to_target': ConfigDict({
            'constructor': go_to_target.GoToTarget,
            'config': ConfigDict(dict(
                moving_target=True
            ))
        }),
        'velocity_control': ConfigDict({
            'constructor': velocity_control.VelocityControl,
            'config': ConfigDict(dict(
                max_speed=4.5,
                reward_margin=0.75,
                direction_exponent=1.,
                steps_before_changing_velocity=166
            ))
        })
    }

    return tasks[task_string]
