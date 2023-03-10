"""Simple floor arenas."""


from .arena import Arena
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
import numpy as np

_GROUNDPLANE_QUAD_SIZE = 0.25


class FloorWithWalls(Arena):

    def _build(self, size=(100, 100), reflectance=.2, aesthetic='default',
               name='floor', top_camera_y_padding_factor=1.1,
               top_camera_distance=100):
        super()._build(name=name)
        self._size = size
        self._top_camera_y_padding_factor = top_camera_y_padding_factor
        self._top_camera_distance = top_camera_distance

        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[.4, .4, .4], diffuse=[.8, .8, .8], specular=[.1, .1, .1])

        # Build groundplane texture.
        self._ground_texture = self._mjcf_root.asset.add(
            'texture',
            rgb1=[.2, .3, .4],
            rgb2=[.1, .2, .3],
            type='2d',
            builtin='checker',
            name='groundplane',
            width=200,
            height=200,
            mark='edge',
            markrgb=[0.8, 0.8, 0.8])
        self._ground_material = self._mjcf_root.asset.add(
            'material',
            name='groundplane',
            texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
            texuniform=True,
            reflectance=reflectance,
            texture=self._ground_texture)

        # Build groundplane.
        self._ground_geom = self._mjcf_root.worldbody.add(
            'geom',
            type='plane',
            name='groundplane',
            material=self._ground_material,
            # size=[1,1,1]
            size=list(size) + [_GROUNDPLANE_QUAD_SIZE]
        )

        # self.mjcf_model.worldbody.add(
        #     'geom',
        #     name='wall1',
        #     type='box',
        #     size=[5, 0.5, 0.5],
        #     pos=[5, 3, 0])
        self.mjcf_model.worldbody.add(
            'geom',
            name='wall2',
            type='box',
            size=[1, 5.5, 0.5],
            pos=[10, 3, 0])
        # self.mjcf_model.worldbody.add(
        #     'geom',
        #     name='wall11',
        #     type='box',
        #     size=[5, 0.5, 0.5],
        #     pos=[15, -7.5, 0])
        self.mjcf_model.worldbody.add(
            'geom',
            name='wall12',
            type='box',
            size=[1, 5.5, 0.5],
            pos=[20, -7.95, 0])
        # self.mjcf_model.worldbody.add(
        #     'geom',
        #     name='wall21',
        #     type='box',
        #     size=[5, 0.5, 0.5],
        #     pos=[25, 3, 0])
        self.mjcf_model.worldbody.add(
            'geom',
            name='wall22',
            type='box',
            size=[1, 5.5, 0.5],
            pos=[30, 3, 0])
        # self.mjcf_model.worldbody.add(
        #     'geom',
        #     name='wall31',
        #     type='box',
        #     size=[5, 0.5, 0.5],
        #     pos=[35, -7.5, 0])
        self.mjcf_model.worldbody.add(
            'geom',
            name='wall32',
            type='box',
            size=[1, 5.5, 0.5],
            pos=[40, -7.5, 0])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='roof_nofloor',
                                      pos=[5.25, 30, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[5.25, 30, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[5.25, 31, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[5.25, 32, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[5.25, 33, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[5.25, 34, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[5.25, 35, 0],
                                      size=[1, 1, 1])

        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 25, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 26, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 27, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 28, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 29, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 30, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 31, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 32, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 33, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 34, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='table',
                                      pos=[-3, 35, 0],
                                      size=[1, 1, 1])

        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='barrier',
                                      pos=[12, 29, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='barrier',
                                      pos=[12, 26, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='barrier',
                                      pos=[12, 32, 0],
                                      size=[1, 1, 1])

        # Choose the FOV so that the floor always fits nicely within the frame
        # irrespective of actual floor size.
        fovy_radians = 2 * np.arctan2(top_camera_y_padding_factor * size[1],
                                      top_camera_distance)
        self._top_camera = self._mjcf_root.worldbody.add(
            'camera',
            name='top_camera',
            pos=[0, 0, top_camera_distance],
            quat=[1, 0, 0, 0],
            fovy=np.rad2deg(fovy_radians))

    @property
    def ground_geoms(self):
        return (self._ground_geom,)

    def regenerate(self, random_state):
        pass

    @property
    def size(self):
        return self._size
