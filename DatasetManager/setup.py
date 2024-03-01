from setuptools import setup, find_packages

setup(
    name='DatasetManager',  # This is the name of your PyPI-package.
    version='0.1',
    url='git@github.com:SonyCSLParis/DatasetManager.git',
    author='Gaetan Hadjeres',
    author_email='gaetan.hadjeres@sony.com',
    license='BSD',
    #packages=['DatasetManager'],
    packages=find_packages(),
    install_requires=['pymongo', 'music21', 'sshtunnel', 'numpy',
                                                   'torch', 'tqdm', 'pretty_midi', 'glob2']
)
