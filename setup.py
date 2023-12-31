from setuptools import setup

setup(name='relastat',
      version='0.1.4',
      packages=['dga'],
      install_requires=[
          'pandas>=1.3.4',
          'numpy>=1.21.4',
          'scipy>=1.7.3',
          'matplotlib>=3.5.0',
          'networkx>=2.6.3'
      ]
      )
