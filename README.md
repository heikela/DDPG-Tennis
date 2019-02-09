# Tennis with DDPG and Population based training

In this repository I attempt to solve a collaborative table tennis Unity environment with DDPG and population based training.

## The Environment

### Summary and dynamics

The environment is a two agent game resembling the game of tennis. Each agent controls a racket and attempts to move it in such a way as to bounce a ball over a net to the other players area without it falling to the ground or flying out of bounds. The ball moves according to inertia and gravity. The environment is two-dimensional in that the rackets and the ball move in one vertical plane (the possible directions of movement are towards the net, up, down and away from the net - there is no sideways movement along the net direction).

In this scenario, the agents collaborate to try to keep the ball in the air.

### Rewards

Each agent receives their own rewards. They receive +0.1 whenever they hit the ball over the net, and a reward of -0.01 whenever they let the ball fall into the ground or hit it out of bounds.

### Actions

The action space for each agent consists of two continuous variables, corresponding to movement on the forward/backward axis and the up/down axis.

### Observations

For each time step, the agents are able to observe 8 continuous variables corresponding to the position and velocity of the ball and the racket. The environment stacks 3 of the most recent observations together, so that observation vector will in total contain 24 values.

### Solving the environment

In this collaborative project, we consider the environment solved when the better score among the two agents exceeds 0.5 when averaged over 100 episodes. If the average over episodes N + 1 to N + 100 fulfils this condition, we say the environment was solved after episode N.

## Dependencies

This project has four groups of dependencies: IPython Jupyter notebooks, Python libraries used by the agent and the learning algorithm, the Unity Environment described above, and the Unity agents python interface.

### Unity Environment

This project is based on a pre-built Unity Environment for the agents. This is supplied by Udacity at the following locations depending on operating system:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Extract this package into the project directory.

### Jupyter Environment

In order to use the code in this repository locally, you need a Jupyter installation. Follow installation instructions at https://jupyter.org/install or get Jupyter as part of a larger environment such as [Anaconda](https://www.anaconda.com/)

Alternatively, if you have access to the Udacity environment set up for this project in the deep reinforcement learning nanodegree, this environment can be used. It also contains the Unity Environment preinstalled.

### Python dependencies for the algorithm, agent and analysis

The solution depends on the following packages:

- numpy
- scipy
- pytorch (0.4.0)
- matplotlib
- jupyter
- pandas

These can be installed with pip / conda.

In practice I installed everything else from Anaconda navigator, and pytorch with

```
conda install pytorch=0.4.1 cuda90 -c pytorch
```

### Unity ML-agents

Additionally, the unity agents package and its dependencies are required for connecting the python-based agent to the unity environment. The pre-built Unity environment is built with Unity-Python API level 0.4, which means we need to find a compatible version of the unity agents to install.

This can be achieved e.g. by choosing a suitable release commit from git, as follows:

```
git clone git@github.com:Unity-Technologies/ml-agents.git
cd ml-agents
git checkout 1ead1ccc2c842bd00a372eee5c4a47e429432712 
cd python
pip install -e .
```

The chosen commit is the one tagged as version 0.4.0b in [the repository's releases page](https://github.com/Unity-Technologies/ml-agents/releases). The final command installs the python dependencies of the ml agents connector.

## How to run the code

The code to define and train the agent is included in the interactive
Python notebook Report.ipynb. To train the model from scratch,
execute each code cell in the notebook in order.
To evaluate a pre-trained model, create an agent and replace
network weights with ones loaded from a checkpoint included
in the repository. If you read the report outside of
an actual Jupyter environment, e.g. in the Github UI,
some of the output from simulation runs is verbose - 
inside Jupyter this becomes a scrollable area with
limited height, leading to better readability.
