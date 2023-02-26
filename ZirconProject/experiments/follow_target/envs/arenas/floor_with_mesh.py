"""Simple floor arenas."""


from .arena import Arena
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
import numpy as np

_GROUNDPLANE_QUAD_SIZE = 0.25


class FloorWithMesh(Arena):

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
        
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='chair',
        #                               pos=[5.25, 30, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='chair',
        #                               pos=[5.25, 31, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='chair',
        #                               pos=[5.25, 32, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='chair',
        #                               pos=[5.25, 33, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='chair',
        #                               pos=[5.25, 34, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='chair',
        #                               pos=[5.25, 35, 0],
        #                               size=[1, 1, 1])

        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 25, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 26, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 27, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 28, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 29, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 30, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 31, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 32, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 33, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 34, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='table',
        #                               pos=[-3, 35, 0],
        #                               size=[1, 1, 1])

        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='barrier',
        #                               pos=[12, 29, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='barrier',
        #                               pos=[12, 26, 0],
        #                               size=[1, 1, 1])
        # self.mjcf_model.worldbody.add('geom',
        #                               type='mesh',
        #                               mesh='barrier',
        #                               pos=[12, 32, 0],
        #                               size=[1, 1, 1])
        
        #Mesh for Roof walls
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='roof_wall1',
                                      pos=[40, 0, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='roof_wall2',
                                      pos=[40, 0, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='roof_wall3',
                                      pos=[40, 0, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='roof_wall4',
                                      pos=[40, 0, 0],
                                      size=[1, 1, 1])
        
        # Mesh of obtacles
        # First row of chairs
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[15, 1, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[15, 0, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[15, -1, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[15, -2, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[15, -3, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[15, -4, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[15, -5, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[15, -6, 0],
                                      size=[1, 1, 1])
        
        # Second row of chairs
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[25, -1, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[25, 0, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[25, 1, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[25, 2, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[25, 3, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[25, 4, 0],
                                      size=[1, 1, 1])
        
        # Third row of barriers
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='barrier',
                                      pos=[33, 1, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='barrier',
                                      pos=[33, -2, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='barrier',
                                      pos=[33, -5, 0],
                                      size=[1, 1, 1])
        
        # Random placement of chairs
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[36, 3, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[38, 1, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[38, 5, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[39, 4, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[41, 2, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[44, 3, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[44, -1, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[43, 0, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[44.5, 0.25, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[45, 4, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[46, 5, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[45.5, 6, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[47, 4, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[48, 3, 0],
                                      size=[1, 1, 1])
        self.mjcf_model.worldbody.add('geom',
                                      type='mesh',
                                      mesh='chair',
                                      pos=[49, 2, 0],
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