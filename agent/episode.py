import torch
import numpy as np
from collections import deque
from environment import env_manager
from utils import utils
import torch.nn.functional as F
import copy


class episode:
    def __init__(self, cfg, env) -> None:
        self.logprob_ = torch.tensor([0])
        self.env = env
        # self.state = None
        self.state = self.reset_env(cfg)
        self.ep_length = 0
        self.penalty = 0 

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
        
        if 'SuperMarioBros' in cfg.env_name:
            self.state = utils.prepare_initial_state(self.state, cfg)
        elif 'MiniGrid-DoorKey' in cfg.env_name:
            self.state = utils.prepare_state(self.state, cfg)

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
                # action_ = torch.argmax(pred.mean(dim=2))
                # action.append(torch.Tensor([action_]).int())

            else:
                for b in range(pred.shape[0]):
                    action_ = torch.argmax(pred)
                    action.append(torch.Tensor([action_]).int())
        return action

    def run(self, 
            nn_model,
            # env, 
            cfg, 
            epochs, epsilon, train_mode=False):
        if cfg.reset_epbuff:
            self.__init__(cfg, self.env)
        if train_mode==False:
            self.state = self.reset_env(cfg)
            self.ep_length = 0

        cfg.epsilon = epsilon
        done = False
        self.success_cnt = 0
        self.cum_reward = 0
        self.loop_count = 0

        self.nn_model = nn_model
        # self.env = env
        # additional_info = None
        # self.state = torch.from_numpy(np.array(self.env.state)).float()
        env_mng = env_manager.Environment(cfg)

        # self.state = self.reset_env(cfg)

        looper = True
        while looper:
            # print(self.loop_count)
            ep_length_ = copy.copy(self.ep_length)
            self.ep_length += 1
            self.loop_count += 1
            if cfg.policy_value_nn:
                # policy, value = self.nn_model(self.state)
                policy, value = self.nn_model(self.state.unsqueeze(dim=0))
                action = self.get_action(policy, cfg)
                self.logprob_ = policy.view(-1)[action]
            elif cfg.policy_nn:
                policy = self.nn_model(self.state)
                action = self.get_action(policy, cfg)
            elif cfg.value_nn:
                # print(self.state.unsqueeze(dim=0).shape)
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
                                                # value/0.5
                                                # value
                                                , dim=1
                                            ), num_samples=1)
                    
                    # if self.ep_length % 100 == 0:
                    #     print('----------------xxxx---------------------')
                    # #     print(value)
                    # #     print(F.normalize(value))
                    # #     print(value/0.5)
                    #     print(F.softmax(F.normalize(value)))
                    # #     print(F.softmax(value/0.5, dim=1))
                    #     print(action)
                    #     print('-------------------------------------')
                        
                else: 
                    action = self.get_action(value.detach(), cfg)
                if isinstance(action, list):
                    action = action[0]
                
                if hasattr(cfg, 'action_map'):
                    action_ = utils.action_map(action,cfg)


            state_next, reward, done, self.info, _ = self.env.step(
                                                int(action_.detach().numpy())
                                                if hasattr(cfg, 'action_map') else
                                                int(action.detach().numpy())
                                                )
            # self.state = torch.from_numpy(state_next).float()
            reward, rep_priority = env_mng.get_reward(done, reward)


            ######################### for mario #############################
            # to make it common using later with queue append
            if 'SuperMarioBros' in cfg.env_name:
                if self.ep_length > 1:
                    additional_info = utils.get_info(cfg, self.info) # Value type
                if not done and hasattr(cfg, 'additional_act_repeats'):
                    for i in range(cfg.additional_act_repeats):
                        additional_info = utils.get_info(cfg, self.info)
                        state_next, reward_, done, self.info, _ = self.env.step(
                                                            int(action.detach().numpy()))
                        reward, rep_priority = env_mng.get_reward(done, reward)     
                        if done:
                            break
                        reward += reward_
                        if hasattr(cfg, 'multi_states_size') and \
                            i < (cfg.additional_act_repeats-1):
                            self.state = utils.prepare_multi_state(self.state, state_next, cfg)
                if hasattr(cfg, 'multi_states_size'):
                    if cfg.multi_states_size > 1:
                        state_next = torch.from_numpy(state_next.copy()).float()
                        state_next = utils.prepare_multi_state(self.state, state_next, cfg)
                if hasattr(cfg, 'max_ep_try') and self.ep_length > cfg.max_ep_try and train_mode:
                    ep_res = utils.eval_ep(cfg, self.info, additional_info) # list
                    if 'SuperMarioBros' in cfg.env_name:
                        penalty, additional_info = ep_res
                        # if reward < 0:
                        #     penalty = True
                        # else:
                        #     penalty = False
                        if penalty:
                            self.penalty += 1
                        else:
                            self.penalty = 0
                        if self.penalty > cfg.penalty_thres:
                            done = True
                        # print(self.ep_length, self.penalty, done)
            #################################################################

            if 'MiniGrid-DoorKey' in cfg.env_name:
                state_next = utils.prepare_state(torch.Tensor(state_next), cfg)

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

            
            if rep_priority:
                # print(f'pass episode! {train_mode}at epochs{epochs} done _ {done} len_ep __ {self.ep_length}')
                self.success_cnt += 1
            if train_mode:
                self.ep_buffer.append(exprience)
            if cfg.exp_priority and rep_priority and train_mode:
                for i in range(cfg.priority_level):
                    self.ep_buffer.append(exprience)


            if done:
                self.state = self.reset_env(cfg)
                self.penalty = 0
                if train_mode:
                    ep_length_ = copy.copy(self.ep_length)
                    self.ep_length = 0
                
            else:
                self.state = torch.tensor(state_next, dtype=torch.float32)
                # if 'MiniGrid-DoorKey' in cfg.env_name:
                #     self.state = utils.prepare_state(self.state, cfg)

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
                else: # len(self.ep_buffer) == cfg.epbuff_size
                    if self.loop_count < cfg.buff_headroom:
                        looper = True
                    else:
                        looper = False
            
            if hasattr(cfg, 'batch_priority') and cfg.batch_priority and train_mode:
                if self.loop_count < cfg.buff_headroom \
                    or len(self.ep_buffer) < cfg.batch_size:
                    looper = True
                else:
                    looper = False
            
            # if self.ep_length % 200 == 0:
            #     print(done, self.ep_length, train_mode, self.env.max_steps, self.env.env.max_steps)
            # self.env.max_steps = cfg.max_steps
            # self.env.env.max_steps = cfg.max_steps


            # print(self.loop_count, len(self.ep_buffer))
            if not train_mode:
                if done:
                    looper = False
                if self.ep_length > 400:
                    looper = False

            # self.env.render()
        results = dict(
                    ep_length=ep_length_,
                    ep_buffer=self.ep_buffer,
                    success_cnt=self.success_cnt,
                    cum_reward=self.cum_reward,
                    )
        
        return results

