from dm_control.viewer import application
from dm_control.viewer import user_input

_INSERT_TARGET = (user_input.MOUSE_BUTTON_LEFT, user_input.MOD_SHIFT)
_SELECT_OBJECT = user_input.DoubleClick(user_input.MOUSE_BUTTON_LEFT)


class Application(application.Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _perform_deferred_reload(self, *args, **kwargs):
        super()._perform_deferred_reload(*args, **kwargs)

        def custom_handler():
            body_id, position = self._viewer.camera.raycast(
                self._viewport, self._viewer._mouse.position)
            if position is not None:
                print(list(position), ',')
            else:
                print(position)
        self._viewer._input_map.bind(custom_handler, _INSERT_TARGET)
