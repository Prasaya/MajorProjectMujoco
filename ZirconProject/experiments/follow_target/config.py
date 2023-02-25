"""
Configs for the training script. Uses the ml_collections config library.
"""
from ml_collections import ConfigDict
from go_to_target import GoToTarget

def get_config(task_string):
    tasks = {
        'go_to_target': ConfigDict({
            'constructor': GoToTarget,
            'config': ConfigDict(dict(
                moving_target=True
            ))
        }),
    }

    return tasks[task_string]
