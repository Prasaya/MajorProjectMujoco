"""Obstacles for agent."""


from dm_control import composer
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
import numpy as np

import os

from dm_control import mjcf
from dm_control.composer import entity as entity_module

_ARENA_XML_PATH = os.path.join(os.path.dirname(__file__), 'obstacles.xml')


class Obstacles(entity_module.Entity):
  """The base empty arena that defines global settings for Composer."""

  def _build(self, name=None):
    """Initializes this arena.

    Args:
      name: (optional) A string, the name of this arena. If `None`, use the
        model name defined in the MJCF file.
    """
    self._mjcf_root = mjcf.from_path(_ARENA_XML_PATH)
    if name:
      self._mjcf_root.model = name
    self.mjcf_model.worldbody.add('geom', 
        type='mesh', 
        mesh='chair', 
        pos=[10.25, +2, 0], 
        size=[1, 1, 1])

  
  @property
  def mjcf_model(self):
    return self._mjcf_root
