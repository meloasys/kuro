import torch, random
import numpy as np
from utils import utils
from tqdm import tqdm

class RLagent:
    def __init__(self, 
                nn_network, optimizer, loss_inst,
                environment, episode,
                configs) -> None:
        self.nn_network = nn_network
        self.optimizer = optimizer
        self.loss_fn = loss_inst
        self.env = environment
        self.ep = episode
        self.cfg = configs
        self.returns = None
        self.target_dist = None
        self.next_values = torch.tensor([0])
        self.epochs = 0

    def test(self):
        avg = 0
        avg_list = []
        for i in tqdm(range(10)):
            results = self.ep.run(
                        self.nn_network,
                        self.env,
                        self.cfg,
                        self.epochs,
                        self.cfg.epsilon,
                        train_mode=False
                        ) # result incl values, logprobs, rewards
            avg = (avg*i + results['ep_length']) / (i+1)
        print("test finished... avg length of episode is ", avg)
        return results
    
    def run_update(self):
        self.epochs += 1
        results = self.ep.run(
                    self.nn_network,
                    self.env,
                    self.cfg,
                    self.epochs,
                    self.cfg.epsilon,
                    ) # result incl values, logprobs, rewards
        loss = self.update_params(results)

        if self.epochs > 100 and self.cfg.epsilon > self.cfg.epsilon_min: #L
            dec = 1./np.log2(self.epochs)
            dec /= 1e3
            self.cfg.epsilon -= dec

        return loss.item()
    

    def update_params(self, results):
        '''
        results includes ep_buffer and others...
        ep_buffer dictionary includes value, action, reward, next_state, logprob
        '''
        self.optimizer.zero_grad()
        ep_buffer = results['ep_buffer']
    
        # add 1. batch size, shuffle, 
        if self.cfg.sampled_batch:
            ep_buffer = random.sample(ep_buffer, self.cfg.batch_size)

        # values = torch.stack([buff['value'] for buff in ep_buffer], dim=1).view(-1)
        values = torch.stack([buff['value'].squeeze(dim=0) for buff in ep_buffer], dim=0)
        actions = torch.tensor([buff['action'] for buff in ep_buffer])
        rewards = torch.tensor([buff['reward'] for buff in ep_buffer])
        # logprobs = torch.stack([buff['logprob'] for buff in ep_buffer], dim=0).view(-1)
        logprobs = torch.stack([buff['logprob'] for buff in ep_buffer], dim=0)
        dones = torch.tensor([buff['done'] for buff in ep_buffer])

        # next_states = torch.stack([torch.tensor(buff['next_state']) for buff in ep_buffer], dim=0)
        states = torch.stack([buff['state'] for buff in ep_buffer], dim=0)
        next_states = torch.stack([buff['next_state'] for buff in ep_buffer], dim=0)


        if self.cfg.flip_res:
            values = values.flip(dims=(0,))
            actions = actions.flip(dims=(0,))
            logprobs = logprobs.flip(dims=(0,))
            rewards = rewards.flip(dims=(0,))
            dones = dones.flip(dims=(0,))
            states.flip(dims=(0,))
            next_states.flip(dims=(0,))
        
        if self.cfg.policy_value_nn:
            self.returns = self.get_returns(dones, values, rewards)
            values = torch.nn.functional.normalize(values, dim=0)
            bathes = (values, logprobs, self.returns)
        elif self.cfg.policy_nn:
            pass # TBD
        elif self.cfg.value_nn:
            # with torch.no_grad():
            values = self.nn_network(states.detach())
            self.next_values = self.nn_network(next_states.detach())
            support = utils.get_support(self.cfg)
            self.target_dist = self.get_target_dist(
                                            self.next_values,
                                            actions,
                                            rewards,
                                            support)
            bathes = (values, self.target_dist.detach())

        # print(values.shape, actions.shape, rewards.shape, logprobs.shape, next_states.shape, self.next_values.shape)
        
        total_loss = self.loss_fn.get_loss(
                                        batches=bathes
                                        )
        
        # total_loss.requires_grad_(True)
        total_loss.backward()
        self.optimizer.step()

        return total_loss

    def get_returns(self, done_batch, value_batch, reward_batch):
        '''for ActorCritic model'''
        assert self.cfg.flip_res, "you should flip episode result to get returns"
        returns = [] # 수익, return - v(s) = 이익
        if self.cfg.n_step_mode:
            if done_batch[0] == 1:
                return_ = torch.Tensor([0])
            else: # bootstrapping 
                return_ = value_batch[0].detach()
        else:
            return_ = torch.Tensor([0])
        for i in range(reward_batch.shape[0]):
            return_ = reward_batch[i] + self.cfg.gamma * return_
            returns.append(return_)  
        returns = torch.stack(returns).view(-1)
        returns = torch.nn.functional.normalize(returns, dim=0)
        return returns
        
        
    def update_dist(self, reward, support, probs):
        '''fn only works for distributed DQN'''
        gamma = self.cfg.dist_gamma
        nsup, vmin, vmax, dz = self.get_deltaz(support)
        bj = np.round((reward - vmin) / dz)
        bj = int(np.clip(bj, 0, nsup-1))
        m = probs.clone()
        j = 1
        for i in range(bj, 1, -1):
            m[i] += np.power(gamma, j) * m[i-1]
            j += 1
        j = 1
        for i in range(bj, nsup-1, 1):
            m[i] += np.power(gamma, j) * m[i+1]
            j += 1
        m /= m.sum()
        return m
    
    def get_target_dist(self, 
                        dist_batch, action_batch, reward_batch, 
                        support,
                        ):
        '''fn only works for distributed DQN'''
        nsup, vmin, vmax, dz = self.get_deltaz(support)
        target_dist_batch = dist_batch.clone()
        for i in range(dist_batch.shape[0]):
            dist_full = dist_batch[i]
            action = int(action_batch[i].item())
            dist = dist_full[action]
            r = reward_batch[i]
            if r != -1: # when episode ends
                target_dist = torch.zeros(nsup)
                bj = np.round((r - vmin) / dz)
                bj = int(np.clip(bj, 0, nsup-1))
                target_dist[bj] = 1.
            else:
                target_dist = self.update_dist(r, support, dist)
            target_dist_batch[i,action,:] = target_dist
        return target_dist_batch

    def get_deltaz(self, support):
        '''fn only works for distributed DQN'''
        nsup = support.shape[0]
        vmin = self.cfg.dist_support_limit[0]
        vmax = self.cfg.dist_support_limit[1]
        dz = (vmax - vmin) / (nsup - 1.)
        return nsup, vmin, vmax, dz
    

