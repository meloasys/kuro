import torch
import numpy as np
from collections import deque
from environment import env_manager
from utils import utils


class episode:
    def __init__(self, cfg) -> None:
        if cfg.n_step_mode:
            cfg.ep_buffer_size = 100000
        self.ep_buffer = deque(maxlen=cfg.ep_buffer_size)
        self.eps = cfg.epsilon
        self.logprob_ = torch.tensor([0])

    def get_action(self, pred, cfg):
        if cfg.policy_value_nn or cfg.policy_nn:
            logits = pred.view(-1)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
        elif cfg.value_nn:
            action = []
            support = utils.get_support(cfg)
            for b in range(pred.shape[0]):
                expectations = [support @ pred[b,a,:] for a in range(pred.shape[1])]  
                action_ = torch.tensor(np.argmax(expectations))
                action.append(action_)
        return action

    def run(self, nn_model, env, cfg, epochs, epsilon, train_mode=True):
        self.__init__(cfg)

        cfg.epsilon = epsilon
        done = False
        j = 0

        self.nn_model = nn_model
        self.env = env
        # self.state = torch.from_numpy(np.array(self.env.state)).float()
        self.state = torch.from_numpy(np.array(self.env.reset()[0])).float()

        looper = True
        while looper:
            j += 1
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
                assert not (cfg.greedy and cfg.softmax), \
                            'you should choose greedy or softmax for exploration'
                if cfg.greedy:
                    if np.random.rand(1) < cfg.epsilon:
                        # action = np.random.randint(cfg.action_space)
                        action = torch.randint(cfg.action_space,(1,))
                    else:
                        action = self.get_action(value.detach(), cfg)
                elif cfg.softmax:
                    # TBD
                    pass
                else:
                    action = self.get_action(value.detach(), cfg)
                if isinstance(action, list):
                    action = action[0]
            
            # print(action)
            state_next, reward, done, info, _ = self.env.step(int(action.detach().numpy()))
            # self.state = torch.from_numpy(state_next).float()

            reward, rep_priority = env_manager.Environment(cfg).get_reward(done, reward)


            exprience = dict(
                        value=value, 
                        action=action, 
                        reward=reward, 
                        state=self.state, 
                        next_state=torch.tensor(state_next, dtype=torch.float32),
                        logprob=self.logprob_,
                        done=done, 
                        )
            
            if train_mode:
                self.ep_buffer.append(exprience)

                if cfg.exp_priority and rep_priority:
                    print(f'pass episode! at {epochs} done _ {done} len_ep __ {j}')
                    for i in range(cfg.priority_level):
                        self.ep_buffer.append(exprience)
            
            self.state = torch.tensor(state_next, dtype=torch.float32)
            
            if done:
                print(f'ep___ done at {j}')
                self.state = torch.from_numpy(np.array(self.env.reset()[0])).float()
            
            del reward

            if cfg.n_step_mode and not cfg.replay_mode and train_mode:
                if done==False and j<cfg.n_step: looper = True
                else: looper = False
            elif not cfg.n_step_mode and cfg.replay_mode and train_mode:
                if len(self.ep_buffer)<cfg.ep_buffer_size: looper = True
                else: looper = False
            else:
                if done==False: looper = True
                else: looper = False

        results = dict(
                    ep_length=j,
                    ep_buffer=self.ep_buffer,
                    )
        
        return results

