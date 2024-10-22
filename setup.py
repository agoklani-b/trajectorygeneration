import os
from glob import glob
from setuptools import setup

package_name = 'traj_gen'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'matplotlib',
        'imageio',
    ],
    zip_safe=True,
    maintainer='agoklani-b',
    maintainer_email='opensource@example.com',
    description='Lightweight 3D voxel A* path planner with a demo script.',
    python_requires='>=3.8',
    license='MIT',
)
