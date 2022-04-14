from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gym_simplegrid',
    version='1.0.2',
    keywords='reinforcement learning, environment, gridworld, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/damat-le/gym-simplegrid',
    description='Simple Gridworld Environment for OpenAI Gym',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['gym_simplegrid', 'gym_simplegrid.envs'],
    install_requires=[
        'gym>=0.23.0',
        'numpy>=1.22.0',
        'matplotlib>=3.5.0'
    ],
    python_requires=">=3.7",
    author="Leo D'Amato",
    author_email="leo.damato.dev@gmail.com",
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"]
)
