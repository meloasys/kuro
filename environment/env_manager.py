import gym

class Environment:
    env: object
    cfg: object
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.rep_priority = False
        self.reward = None

    def get_reward(self, done, reward):
        if self.cfg.env_type=='gym':
            if self.cfg.env_name=='CartPole-v1':
                if done:
                    self.reward = -10.0
                    self.env.reset()
                else:
                    self.reward = 1.0
                    self.rep_priority = True
            elif self.cfg.env_name=='Freeway-ram-v0':
                # print(done, reward)
                if done:
                    self.reward = -10
                elif reward == 1:
                    self.reward = 10
                    self.rep_priority = True
                elif reward == 0:
                    self.reward = -1
                else:
                    self.reward = reward

        return self.reward, self.rep_priority

    @classmethod
    def set_env(cls, cfg) -> None:
        cls.env = None
        cls.cfg = cfg
        if cls.cfg.env_type=='gym':
            cls.env = gym.make(cls.cfg.env_name)
        if cls.cfg.env_type=='custom':
            print('[info] You choose custom environment')
    
    @classmethod
    def get_env(cls):
        return cls.env


            