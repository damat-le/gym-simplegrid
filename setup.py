from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='gym_simplegrid',
    version='1.1.0',
    keywords='reinforcement learning, environment, gridworld, agent, rl, gymnasium, farama-foundation',
    url='https://github.com/damat-le/gym-simplegrid',
    description='Simple Gridworld Environment for Gymnasium',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['gym_simplegrid', 'gym_simplegrid.*']),
    include_package_data=True,
    install_requires=[
        'gymnasium>=0.26.0',
        'numpy>=1.20.0',
        'matplotlib>=3.5.0'
    ],
    python_requires=">=3.9",
    author="Leo D'Amato",
    author_email="leo.damato.dev@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/damat-le/gym-simplegrid/issues",
        "Source Code": "https://github.com/damat-le/gym-simplegrid",
    },
)