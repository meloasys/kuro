import torch
import numpy as np

class episode:
    def __init__(self) -> None:
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.done = False
        self.j = 0
        self.G = torch.Tensor([0])
    
    # def run(self):
    def run(self, nn_model, env, cfg, train_mode=True):
        self.__init__()
        self.nn_model = nn_model
        self.env = env
        self.state = torch.from_numpy(np.array(self.env.state)).float()
        while ((self.done==False and self.j<cfg.n_step) 
               if (cfg.n_step_mode and train_mode)
               else self.done==False):
            self.j += 1
            policy, value = self.nn_model(self.state)
            self.values.append(value)
            logits = policy.view(-1)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            logprob_ = policy.view(-1)[action]
            self.logprobs.append(logprob_)
            state_next, _, self.done, info, _ = self.env.step(action.detach().numpy())
            self.state = torch.from_numpy(state_next).float()
            if self.done:
                reward = -10.0
                self.env.reset()
                self.G = torch.Tensor([0])
            else:
                reward = 1.0
                self.G = value.detach()
            self.rewards.append(reward)
        results = dict(
                    values=self.values, 
                    logprobs=self.logprobs, 
                    rewards=self.rewards, 
                    ep_length=self.j,
                    G=self.G)
        return results

