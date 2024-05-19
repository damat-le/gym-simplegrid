from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gym_simplegrid',
    version='1.0.6',
    keywords='reinforcement learning, environment, gridworld, agent, rl, openaigym, openai-gym, gym, gymnasium, farama-foundation',
    url='https://github.com/damat-le/gym-simplegrid',
    description='Simple Gridworld Environment for Gymnasium',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['gym_simplegrid', 'gym_simplegrid.envs'],
    install_requires=[
        'gymnasium',
        'matplotlib'
    ],
    python_requires=">=3.7",
    author="Leo D'Amato",
    author_email="leo.damato.dev@gmail.com",
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"]
)
