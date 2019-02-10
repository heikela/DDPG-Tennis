import re


class OpponentFinder():
    """A class that helps us determine which agent played against which on each episode"""
    def __init__(self, filename):
        """Construct the OpponentFinder based on a training log"""
        self.opponent_directory = [dict() for i in range(16)]
        lines = open(filename, "r").readlines()
        agents = []
        agent_pattern = re.compile(r"Paired agent-(\d+) with agent-(\d+)")
        episode_pattern = re.compile(r"Episode (\d+)")
        for line in lines:
            agent_match = agent_pattern.match(line)
            if agent_match:
                agents = [int(agent_match.group(1)), int(agent_match.group(2))]
            elif len(agents) > 0:
                episode_match = episode_pattern.match(line)
                if episode_match:
                    episode = int(episode_match.group(1))
                    self.opponent_directory[agents[0]][episode] = agents[1]
                    self.opponent_directory[agents[1]][episode] = agents[0]
                    agents = []

    def find_opponent(self, agent, episode):
        lookup_episode = 200 * ((episode + 199) // 200)
        return self.opponent_directory[agent][lookup_episode]
