from distutils.core import setup

setup(
    name='MajorProjectMujco',
    version='0.1',
    author='Prasaya Acharya',
    author_email='prasaya.acharya@gmail.com',
    license='MIT',
    packages=['MajorProjectMujco'],
    install_requires=[
        'azure.storage.blob==12.9.0',
        'cloudpickle>=2.1.0',
        'gym==0.21',
        'h5py',
        'imageio',
        'imageio-ffmpeg',
        'ml_collections',
        'mujoco',
        'pytorch-lightning<1.7',
        'stable-baselines3'
    ]
)