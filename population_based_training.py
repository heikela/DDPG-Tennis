import numpy as np
from scipy import stats
import torch
import agent
import os

def population_based_training(agents, env, brain_name,
                              series_rounds=20,
                              rounds_between_exploit=1,
                              max_t=1100,
                              p_threshold = 0.05,
                              checkpoint_episodes = None,
                              result_dir='.'):
    """
    Implement population based training of neural networks in the context of DDPG
    
    Params
    ======
        agents (list(DdpgAgent)): The population of agents to train
        env (Unity Env): The environment to train the agent in
        brain_name (string): The name of the brain to use when communicating to the unity env.
        series_rounds (int): How many full series of everyone against everyone in both positions to carry out overall.
        rounds_between_exploit (int): How many full rounds of everyone against everyone in both positions to carry out
            between exploits.
        max_t (int): Maximum number of time steps to allow in each episode if the environment doesn't indicate
            that the episode is done sooner than this.
        episodes_between_exploit (int): conduct the exploit step (and potentially the explore step) every this many episodes
        p_threshold (float): the threshold for the T-test p-value below which
            the exploit step will overwrite one agent with antoher
        checkpoint_episodes (int): after how many episodes we want to save a snapshot of the agent's state on disk.
            If not given, defaults to episodes_between_exploit
    """
    episodes_between_exploit = rounds_between_exploit * (len(agents) - 1)
    checkpoint_episodes = checkpoint_episodes if checkpoint_episodes else episodes_between_exploit
    exploit_round = len(agents[0].history) // episodes_between_exploit
    for round in range(0, series_rounds):
#        episode_i += episodes_between_exploit
#        randomized_agent_idx = np.random.permutation(len(agents))
        for delta in range(1, len(agents)):
            for agent1_idx in range(0, len(agents)):
                agent2_idx = (agent1_idx + delta) % len(agents)
                agent_pair = [agents[agent1_idx], agents[agent2_idx]]
                agent.ddpg_collab(
                    agent_pair, env, brain_name,
                    episodes=1,
                    checkpoint_episodes=checkpoint_episodes,
                    result_dir=result_dir
                )
    #            for i in range(0,2):
    #                print("\r\nAgent: {}, mean return: {:.2f}".format(agent_pair[i].name, agent_pair[i].get_running_mean_return(episodes_between_exploit)))
        new_exploit_round = len(agents[0].history) // episodes_between_exploit
        if new_exploit_round != exploit_round:
            for target_i, target in enumerate(agents):
                # uniformly select a candidate that is not the same as target
                candidate_i = np.random.randint(0, len(agents) - 1)
                if candidate_i >= target_i:
                    candidate_i += 1
                candidate = agents[candidate_i]
                # exploit
                candidate_scores = candidate.history.tail(episodes_between_exploit)['return']
                target_scores = target.history.tail(episodes_between_exploit)['return']
                exploited = False
                if candidate_scores.mean() > target_scores.mean():
                    statistic, p = stats.ttest_ind(candidate_scores, target_scores, equal_var=False)
                    if p < p_threshold:
                        print("Overwriting {} with {}, p = {:.2f}".format(target.name, candidate.name, p))
                        agent_name = target.name
                        torch.save(target.full_save_dict(),
                                  os.path.join(result_dir, "retire_{}_episode_{}.pth".format(agent_name, len(target.history))))
                        exploited = True
                        target.load_state_dict(candidate.state_dict())
                    else:
                        print("{} performed worse than {} but it wasn't significant, p = {:.2f}".format(target.name, candidate.name, p))
                if exploited:
                    # explore
                    source_hyperparameters = candidate.hyperparameter_dict()
                    target.load_mutated_hyperparameter_dict(source_hyperparameters)
            exploit_round = new_exploit_round

