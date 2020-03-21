from gym import Env
from agents import Agent

class MultiEnv(Env):

    r"""
    A layer over the Gym Env class able to handle environements with multiple agents.
    
    The main add in MultiEnv is the method
        turn: returns the next agent to play given the state
    And the attributes :
        n_players: number of players in the environement
        active_player: id of the active player

    On top of the main API basic methodes
        step: take a step of the environement given the action of the active player
        reset  
        render  
        close  
        seed

    And attributes:

        action_space: The Space object corresponding to valid actions  
        observation_space: The Space object corresponding to valid observations  
        reward_range: A tuple corresponding to the min and max possible rewards  

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    """

    # This must be set in all MultiEnv subclasses
    n_players = None
    first_active_player = 0

    def turn(self, state):
        raise NotImplementedError 


class Playground():

    def __init__(self, environement:Env, agents):
        assert isinstance(environement, Env)
        if isinstance(agents, Agent):
            agents = [agents]
        for agent in agents:
            assert isinstance(agent, Agent)
        
        self.env = environement
        self.agents = agents

    def fit(self, episodes, render=False, learn=True, verbose=0):
        print_cycle = max(1, episodes // 100)
        avg_gain = 0
        for episode in range(episodes):

            state = self.env.reset()
            done = False
            gain = 0

            while not done:

                if render: self.env.render()

                if isinstance(self.env, MultiEnv):
                    agent = self.agents[self.env.turn(state)]
                else:
                    agent = self.agents[0]

                action = agent.act(state)
                next_state, reward, done , info = self.env.step(action)
                gain += reward
                
                agent.remember(state, action, reward, done, next_state, info)
                state = next_state

                if learn: agent.learn()
            
            if verbose > 0:
                avg_gain += gain
                if episode%print_cycle==0: 
                    print(f'Episode {episode}/{episodes}, Average return over {print_cycle} eps:{avg_gain/print_cycle}')
                    avg_gain = 0

    def run(self, episodes, verbose=0):
        self.fit(episodes, render=True, learn=False, verbose=verbose)
