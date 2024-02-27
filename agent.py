import torch
import numpy as np

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
        self.cfg=configs

    def run_update(self):
        self.env.reset()
        results = self.ep.run(
                    self.nn_network,
                    self.env,
                    self.cfg,
                    ) # result incl values, logprobs, rewards
        
        # losses = self.update_params(
        #                     optimizer=self.optimizer,
        #                     results=results,
        #                     flip_res=self.cfg.flip_res,
        #                     n_step_mode=self.cfg.n_step_mode,
        #                     )

        losses = self.update_params2(results)

        return losses
    
    def test(self):
        import numpy as np

        avg = 0
        avg_list = []
        for i in range(10):
            self.env.reset()
            results = self.ep.run(
                        self.nn_network,
                        self.env,
                        self.cfg,
                        train_mode=False
                        ) # result incl values, logprobs, rewards
            avg = (avg*i + results['ep_length']) / (i+1)
            
        # print("test finished... avg length of episode is ",results['ep_length'])
        print("test finished... avg length of episode is ", avg)
        return results

    def update_params(self, optimizer, results, 
                      clc=0.1, gamma=0.9,
                      **kwagrs):
        values = torch.stack(results['values']).view(-1)
        logprobs = torch.stack(results['logprobs']).view(-1)
        if not isinstance(results['rewards'][0], torch.Tensor): 
            rewards = torch.Tensor(results['rewards']).view(-1)
        if kwagrs['flip_res']:
            values = values.flip(dims=(0,))
            logprobs = logprobs.flip(dims=(0,))
            rewards = rewards.flip(dims=(0,))

        returns = [] # 수익, return - v(s) = 이익
        if kwagrs['n_step_mode']:
            return_ = results['G']
        else:
            return_ = torch.Tensor([0])
        for i in range(rewards.shape[0]):
            return_ = rewards[i] + gamma * return_
            returns.append(return_)
        returns = torch.stack(returns).view(-1)
        returns = torch.nn.functional.normalize(returns, dim=0)
        # better result for q_val normalized
        values = torch.nn.functional.normalize(values, dim=0)
        bathes = (values, logprobs, returns)
        actor_loss, critic_loss, total_loss = self.loss_fn.get_loss(
                                                        batches=bathes,
                                                        weight=clc,
                                                        )
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return (actor_loss, critic_loss, total_loss)

    def update_params2(self, results):
        '''
        results includes ep_buffer and others...
        ep_buffer dictionary includes value, action, reward, next_state, logprob
        '''
        ep_buffer = results['ep_buffer']
    
        # add 1. batch size, shuffle, 

        values = torch.stack([buff['value'] for buff in ep_buffer], dim=1).view(-1)
        actions = torch.tensor([buff['action'] for buff in ep_buffer])
        rewards = torch.tensor([buff['reward'] for buff in ep_buffer])
        logprobs = torch.stack([buff['logprob'] for buff in ep_buffer], dim=0).view(-1)
        dones = torch.tensor([buff['done'] for buff in ep_buffer])

        next_states = torch.stack([torch.tensor(buff['next_state']) for buff in ep_buffer], dim=0)

        if self.cfg.flip_res:
            values = values.flip(dims=(0,))
            logprobs = logprobs.flip(dims=(0,))
            rewards = rewards.flip(dims=(0,))
            dones = dones.flip(dims=(0,))
        
        # print(values.shape, actions.shape, rewards.shape, logprobs.shape, next_states.shape, dones)
            
        returns = [] # 수익, return - v(s) = 이익
        
        if self.cfg.n_step_mode:
            if dones[0] == 1:
                return_ = torch.Tensor([0])
            else:
                return_ = values[0].detach()
        else:
            return_ = torch.Tensor([0])
        
        for i in range(rewards.shape[0]):
            return_ = rewards[i] + self.cfg.gamma * return_
            returns.append(return_)
            
        returns = torch.stack(returns).view(-1)
        returns = torch.nn.functional.normalize(returns, dim=0)
        # better result for q_val normalized
        values = torch.nn.functional.normalize(values, dim=0)

        # print(values, logprobs, returns)

        bathes = (values, logprobs, returns)
        
        actor_loss, critic_loss, total_loss = self.loss_fn.get_loss(
                                                        batches=bathes,
                                                        weight=self.cfg.clc,
                                                        )
        
        # total_loss.requires_grad_(True)
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return (actor_loss, critic_loss, total_loss)

        
        
    def update_dist(self, reward, support, probs):
        '''fn only works for distributed DQN'''
        gamma = self.cfg['dist_gamma']
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
        vmin = self.cfg['dist_support_limit'][0]
        vmax = self.cfg['dist_support_limit'][1]
        dz = (vmax - vmin) / (nsup - 1.)
        return nsup, vmin, vmax, dz
    
    def get_action(self, dist, support):
        actions = []
        for b in range(dist.shape[0]):
            expectations = [support @ dist[b,a,:] for a in range(dist.shape[1])]
            action = int(np.argmax(expectations))
            actions.append(action)
        actions = torch.Tensor(actions).int()
        return actions

    
    

