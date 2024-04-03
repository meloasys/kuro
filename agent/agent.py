import torch, random, copy
import numpy as np
from utils import utils
from tqdm import tqdm

class RLagent:
    def __init__(self, 
                nn_network, optimizer, loss_inst,
                environment, episode,
                configs) -> None:
        self.nn_other_net = None
        self.nn_network = None
        if len(configs.network_models) > 1:
            self.nn_network = nn_network[configs.network_models[0]]
            nn_network.pop(configs.network_models[0])
            self.nn_other_net = nn_network
            print('Agent choose multi network')
        else:
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
        self.eplens_avg = []
        self.successes_avg = []
        self.rewards_avg = []
        if self.cfg.target_nn:
            self.nn_network_t = copy.deepcopy(nn_network)
            # print(self.nn_network_t.state_dict())

    def test(self):
        eplen_avg = 0
        success_avg = 0
        reward_avg = 0
        for i in tqdm(range(self.cfg.test_qty)):
        # for i in range(2):
            results = self.ep.run(
                        self.nn_network,
                        # self.env,
                        self.cfg,
                        self.epochs,
                        self.cfg.epsilon,
                        train_mode=False
                        ) # result incl values, logprobs, rewards
            eplen_avg = (eplen_avg*i + results['ep_length']) / (i+1)
            success_avg = (success_avg*i + results['success_cnt']) / (i+1)
            reward_avg = (reward_avg*i + results['cum_reward']) / (i+1)
        print(f"Test finished..., epoch {self.epochs}")
        print("Avg length of episode is _____ ", eplen_avg)
        print("Avg success of episode is _____ ", success_avg)
        print("Avg reward of episode is _____ ", reward_avg)
        print("epsilon is _____ ", self.cfg.epsilon)
        self.eplens_avg.append(eplen_avg)
        self.successes_avg.append(success_avg)
        self.rewards_avg.append(reward_avg)
        results = dict(
                    eplens_avg=self.eplens_avg, 
                    successes_avg=self.successes_avg, 
                    rewards_avg=self.rewards_avg
                    )
        return results
    
    def run_update(self):
        self.epochs += 1
        results = self.ep.run(
                    self.nn_network,
                    # self.env,
                    self.cfg,
                    self.epochs,
                    self.cfg.epsilon,
                    train_mode=True
                    ) # result incl values, logprobs, rewards
        loss = self.update_params(results)

        if self.epochs > self.cfg.epsilon_thrs \
                and self.cfg.epsilon > self.cfg.epsilon_min: #L
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
            if len(ep_buffer) > self.cfg.batch_size: 
                ep_buffer = random.sample(ep_buffer, self.cfg.batch_size)

        # values = torch.stack([buff['value'] for buff in ep_buffer], dim=1).view(-1)
        values = torch.stack([buff['value'].squeeze(dim=0) for buff in ep_buffer], dim=0)
        actions = torch.tensor([buff['action'] 
                                for buff in ep_buffer])
        rewards = torch.tensor([buff['reward'] 
                                for buff in ep_buffer])
        logprobs = torch.stack([buff['logprob'] 
                                for buff in ep_buffer], dim=0)
        dones = torch.tensor([buff['done'] 
                              for buff in ep_buffer])
        states = torch.stack([buff['state'] 
                              for buff in ep_buffer], dim=0)
        next_states = torch.stack([buff['next_state'] 
                                   for buff in ep_buffer], dim=0)

        if self.cfg.flip_res:
            values = values.flip(dims=(0,))
            actions = actions.flip(dims=(0,))
            logprobs = logprobs.flip(dims=(0,))
            rewards = rewards.flip(dims=(0,))
            dones = dones.flip(dims=(0,))
            states.flip(dims=(0,))
            next_states.flip(dims=(0,))
        
        batch_pack = (values, actions, logprobs, rewards, dones, states, next_states)

        if self.cfg.policy_value_nn:
            self.returns = self.get_returns(dones, values, rewards)
            values = torch.nn.functional.normalize(values, dim=0)
            train_batch = (values, logprobs, self.returns)
        elif self.cfg.policy_nn:
            pass # TBD
        elif self.cfg.value_nn:
            train_batch = self.make_batch(batch_pack)
        # print(values.shape, actions.shape, rewards.shape, logprobs.shape, next_states.shape, self.next_values.shape)

        total_loss = self.loss_fn.get_loss(
                                        batches=train_batch,
                                        loss_type=self.cfg.loss_fn,
                                        )
        # total_loss.requires_grad_(True)
        total_loss.backward()
        self.optimizer.step()

        if self.cfg.target_nn \
                and (self.epochs % self.cfg.t_update == 0):
            self.nn_network_t.load_state_dict(
                                    self.nn_network.state_dict())
        return total_loss

        
    def make_batch(self, batch_pack):
        values, actions, logprobs, rewards, dones, states, next_states = batch_pack
        if self.cfg.nn_mod == 'dist_dqn':
            # stable learning than using forward passed values in episode
            values = self.nn_network(states.detach())
            if self.cfg.target_nn: 
                self.next_values = self.nn_network_t(next_states.detach())
            else:
                self.next_values = self.nn_network(next_states.detach())
            support = utils.get_support(self.cfg)
            self.target_dist = self.get_target_dist(
                                        self.next_values.detach(),
                                        actions,
                                        rewards,
                                        support)
            train_batch = (values, self.target_dist.detach())
        elif self.cfg.nn_mod == 'icm':
            encoder = self.nn_other_net[self.cfg.network_models[1]]
            inverse_model = self.nn_other_net[self.cfg.network_models[2]]
            forward_model = self.nn_other_net[self.cfg.network_models[3]]

            # print(torch.sum(list(inverse_model.parameters())[0]))

            state1_hat = encoder(states)
            state2_hat = encoder(next_states)
            state2_hat_pred = forward_model(state1_hat.detach(), actions.detach())
            pred_action = inverse_model(state1_hat, state2_hat)
            qvals = self.nn_network(states)
            qvals_next = self.nn_network(next_states)

            train_batch = (states, next_states, state1_hat, state2_hat, state2_hat_pred, 
                           actions, pred_action, qvals, qvals_next, rewards)
            
        elif self.cfg.nn_mod == 'multihead_attn':
            qvals = self.nn_network(next_states)
            astar = torch.argmax(qvals, dim=1)
            qvals_t = self.nn_network_t(next_states.detach())
            qs = qvals_t.gather(dim=1, 
                                index=astar.unsqueeze(dim=1)
                                ).squeeze()

            targets = torch.tensor(rewards + (1-dones.numpy()) * self.cfg.gamma,
                                   dtype=torch.float32) * qs.detach()

            train_batch = (targets, qvals, actions)

        return train_batch


    def get_returns(self, done_batch, value_batch, reward_batch):
        '''for ActorCritic model'''
        assert self.cfg.flip_res, \
                "you should flip episode result to get returns"
        returns = [] # 수익, return - v(s) = 이익
        if self.cfg.ep_lim:
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

            expectations = [support @ dist_batch[i,a,:] 
                                for a in range(dist_batch.shape[1])]
            next_q_val = np.max(expectations)
            
            if r != -1: # when episode ends
                target_dist = torch.zeros(nsup)
                bj = np.round((r - vmin) / dz)
                bj = int(np.clip(bj, 0, nsup-1))
                target_dist[bj] = 1.
            else:
                if self.cfg.bootstrap:
                    r = r + self.cfg.dist_gamma*next_q_val
                target_dist = self.update_dist(r, support, dist)
            target_dist_batch[i,action,:] = target_dist
        return target_dist_batch
        
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
    

    def get_deltaz(self, support):
        '''fn only works for distributed DQN'''
        nsup = support.shape[0]
        vmin = self.cfg.dist_support_limit[0]
        vmax = self.cfg.dist_support_limit[1]
        dz = (vmax - vmin) / (nsup - 1.)
        return nsup, vmin, vmax, dz

