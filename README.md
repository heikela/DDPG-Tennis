# DDPG-Tennis

Attempt to solve a collaborative table tennis Unity environment with DDPG and population based training

## Dependencies

### Python dependencies for solution code

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
