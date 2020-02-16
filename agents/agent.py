class Agent():

    state_values = None
    action_values = None
    memory = None

    def policy(self, observation):
        raise NotImplementedError

    def play(self, observation):
        raise NotImplementedError
    
    def remember(self, observation, action, reward, done, next_observation=None, infos={}):
        raise NotImplementedError
    
    def learn(self):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError