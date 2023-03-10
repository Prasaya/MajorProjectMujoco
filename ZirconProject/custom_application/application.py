from dm_control.viewer import application
from dm_control.viewer import user_input
from ZirconProject.custom_application.viewer import Viewer

_INSERT_TARGET = (user_input.MOUSE_BUTTON_LEFT, user_input.MOD_SHIFT)
_SELECT_OBJECT = user_input.DoubleClick(user_input.MOUSE_BUTTON_LEFT)


class Application(application.Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callbacks = {}
        # self._viewer = Viewer(self._viewport, self._window.mouse, self._window.keyboard, pause_subject = self._pause_subject)

    def register_callback(self, callback, keybinding):
        self.callbacks[keybinding] = callback

    def _perform_deferred_reload(self, *args, **kwargs):
        super()._perform_deferred_reload(*args, **kwargs)
        for keybinding, callback in self.callbacks.items():
            self._input_map.bind(callback, keybinding)
