import torch
import numpy as np
from collections import deque

class episode:
    def __init__(self) -> None:
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.done = False
        self.j = 0
        self.G = torch.Tensor([0])
    
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

class episode2:
    def __init__(self, cfg) -> None:
        self.ep_buffer = deque(maxlen=cfg.ep_buffer_size)

    
    def run(self, nn_model, env, cfg, train_mode=True):
        self.__init__(cfg)

        values = []
        logprobs = []
        rewards = []
        done = False
        j = 0

        self.nn_model = nn_model
        self.env = env
        self.state = torch.from_numpy(np.array(self.env.state)).float()
        while ((done==False and j<cfg.n_step) 
               if (cfg.n_step_mode and train_mode)
               else done==False):
            j += 1
            policy, value = self.nn_model(self.state)
            values.append(value)
            logits = policy.view(-1)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            logprob_ = policy.view(-1)[action]
            logprobs.append(logprob_)
            state_next, _, done, info, _ = self.env.step(action.detach().numpy())
            self.state = torch.from_numpy(state_next).float()
            if done:
                reward = -10.0
                self.env.reset()
                g = torch.Tensor([0])
            else:
                reward = 1.0
                # _, value = self.nn_model(self.state)
                g = value.detach()
            rewards.append(reward)

            exprience = dict(
                        value=value, 
                        action=action, 
                        reward=reward, 
                        next_state=state_next, 
                        logprob=logprob_,
                        done=done
                        )
            self.ep_buffer.append(exprience)

        results = dict(
                    values=values, 
                    logprobs=logprobs, 
                    rewards=rewards, 
                    ep_length=j,
                    G=g,
                    ep_buffer=self.ep_buffer,
                    )
        
        return results

