from setuptools import setup, find_packages

setup(
    name='trajectory-privacy-protection',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project to protect trajectory privacy in location-based services by generating realistic, privacy-preserving fake trajectories.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'folium',
        'geopandas',
        'shapely',
        'numpy',
        'pandas',
        'osmnx',
        'pyproj',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)