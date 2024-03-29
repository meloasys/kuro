import torch
import numpy as np
from collections import deque
from environment import env_manager
from utils import utils
import torch.nn.functional as F


class episode:
    def __init__(self, cfg) -> None:
        self.logprob_ = torch.tensor([0])
        self.success_cnt = 0
        self.cum_reward = 0
        self.env = None
        self.state = None

        if cfg.ep_lim or cfg.ignore_done:
            self.ep_buffer = deque(maxlen=cfg.epbuff_size)
        else:
            self.ep_buffer = []
        if hasattr(cfg, 'multi_states_size'):
            self.multi_states = deque(maxlen=cfg.multi_states_size)

    def reset_env(self, cfg):
        env_reset = self.env.reset()
        if isinstance(env_reset, tuple):
            env_reset = env_reset[0]
        self.state = torch.from_numpy(np.array(env_reset)).float()

        # for name, layer in enumerate(self.nn_model.named_children()):
        #     if isinstance(layer, torch.nn.modules.conv):
        #     
        #     break
        
        if cfg.env_name == 'SuperMarioBros':
            self.state = utils.prepare_initial_state(self.state, cfg)

        return self.state

    def get_action(self, pred, cfg):
        if cfg.policy_value_nn or cfg.policy_nn:
            logits = pred.view(-1)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
        elif cfg.value_nn:
            action = []
            if cfg.prob_q:
                support = utils.get_support(cfg)
                for b in range(pred.shape[0]):
                    expectations = [support @ pred[b,a,:] 
                                        for a in range(pred.shape[1])]  
                    action_ = torch.tensor(np.argmax(expectations))
                    action.append(action_)
            else:
                for b in range(pred.shape[0]):
                    action_ = torch.argmax(pred)
                    action.append(action_)
        return action

    def run(self, 
            nn_model,
            env, 
            cfg, 
            epochs, epsilon, train_mode=False):
        if cfg.reset_epbuff:
            self.__init__(cfg)

        cfg.epsilon = epsilon
        done = False
        self.j = 0

        self.nn_model = nn_model
        self.env = env
        additional_info = None
        # self.state = torch.from_numpy(np.array(self.env.state)).float()
        env_mng = env_manager.Environment(cfg)

        self.state = self.reset_env(cfg)

        looper = True
        while looper:
            self.j += 1
            if cfg.policy_value_nn:
                # policy, value = self.nn_model(self.state)
                policy, value = self.nn_model(self.state.unsqueeze(dim=0))
                action = self.get_action(policy, cfg)
                self.logprob_ = policy.view(-1)[action]
            elif cfg.policy_nn:
                policy = self.nn_model(self.state)
                action = self.get_action(policy, cfg)
            elif cfg.value_nn:


                value = self.nn_model(self.state.unsqueeze(dim=0))
                # print(self.state.shape, value.shape)
                if cfg.policy == 'greedy':
                    if np.random.rand(1) < cfg.epsilon:
                        action = torch.randint(cfg.action_space,(1,))
                    else:
                        action = self.get_action(value.detach(), cfg)
                elif cfg.policy == 'softmax':
                    # TBD
                    pass
                elif cfg.policy == 'mix':
                    if epochs > int(cfg.epochs*cfg.greedy_thr_rate):
                        if np.random.rand(1) < cfg.epsilon:
                            action = torch.randint(cfg.action_space,(1,))
                        else:
                            action = self.get_action(value.detach(), cfg)
                    else:
                        action = torch.multinomial(
                                            F.softmax(
                                                F.normalize(value)
                                            ), num_samples=1)
                else: 
                    action = self.get_action(value.detach(), cfg)
                if isinstance(action, list):
                    action = action[0]
            if self.j > 1:
                additional_info = utils.get_info(cfg, info) # Value type
            state_next, reward, done, info, _ = self.env.step(
                                                int(action.detach().numpy()))
            # self.state = torch.from_numpy(state_next).float()
            reward, rep_priority = env_mng.get_reward(done, reward)

            for i in range(cfg.additional_act_repeats):
                additional_info = utils.get_info(cfg, info)
                state_next, reward_, done, info, _ = self.env.step(
                                                    int(action.detach().numpy()))
                reward, rep_priority = env_mng.get_reward(done, reward)     
                if done:
                    self.state = self.reset_env(cfg)
                    break
                reward += reward_
                

            #########################$#@$#@$#@$@#$#$#############################
            if hasattr(cfg, 'multi_states_size'):
                if cfg.multi_states_size > 1:
                    # print(state_next)
                    state_next = torch.from_numpy(state_next.copy()).float()
                    state_next = utils.prepare_multi_state(self.state, state_next, cfg)
 
            #########################$#@$#@$#@$@#$#$#############################

            exprience = dict(
                        value=value, 
                        action=action, 
                        reward=reward, 
                        state=self.state,
                        next_state=torch.tensor(state_next, 
                                                dtype=torch.float32),
                        logprob=self.logprob_,
                        done=done, 
                        )
            
            if self.j > cfg.max_ep_try:
                ep_res = utils.eval_ep(cfg, info, additional_info) # list
                if cfg.env_name == 'SuperMarioBros':
                    done, additional_info = ep_res
            

            
            if rep_priority:
                # print(f'pass episode! at epochs{epochs} done _ {done} len_ep __ {j}')
                self.success_cnt += 1
            if train_mode:
                self.ep_buffer.append(exprience)
            if cfg.exp_priority and rep_priority and train_mode:
                for i in range(cfg.priority_level):
                    self.ep_buffer.append(exprience)


            if done:
                self.state = self.reset_env(cfg)
                self.j = 0
            else:
                self.state = torch.tensor(state_next, dtype=torch.float32)

            self.cum_reward += reward
            del reward
            
            # stop when done or full of buffer
            if cfg.ep_lim and train_mode:
                if done==False and len(self.ep_buffer)<cfg.epbuff_size: 
                    looper = True
                else:
                    looper = False
            else:
                if done==False: 
                    looper = True
                else: 
                    looper = False

            # stop only when full of buffer
            if cfg.ignore_done and train_mode:
                if len(self.ep_buffer)<cfg.epbuff_size:
                    looper = True
                else:
                    looper = False

        results = dict(
                    ep_length=self.j,
                    ep_buffer=self.ep_buffer,
                    success_cnt=self.success_cnt,
                    cum_reward=self.cum_reward,
                    )
        
        return results

