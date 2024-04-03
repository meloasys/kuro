import torch
from torch import nn
import torch.nn.functional as F

class Loss:
    def __init__(self) -> None:
        self.name = 'loss_fn_parents'
    def get_name(self):
        print(self.name)

class LossFn(Loss):
    def __init__(self, config):
        super(LossFn,self).__init__()
        self.cfg = config
        self.mse_non_reduction = nn.MSELoss(reduction='none')
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')



    def get_loss(self, batches, loss_type, *args):

        if loss_type == 'actor-critic':
            values, logprobs, returns = batches
            actor_loss = -1 * logprobs * (returns - values.detach())
            critic_loss = torch.pow(values - returns, 2)
            total_loss = (1-self.cfg.clc)*actor_loss.sum() + self.cfg.clc*critic_loss.sum()         
            return total_loss
        
        elif loss_type == 'dist-cross-entropy':
            '''works for distributed DQN'''
            x, y = batches
            loss = torch.Tensor([0.])
            loss.requires_grad = True
            for i in range(x.shape[0]):
                loss_ = -1 * torch.log(x[i].flatten(start_dim=0)) @ y[i].flatten(start_dim=0)
                loss = loss + loss_
            return loss
        
        elif loss_type == 'icm':
            states, next_states, state1_hat, state2_hat, state2_hat_pred, actions, pred_action, qvals, qvals_next, rewards_ = batches

            _forward_err = self.cfg.forward_scale \
                            * self.get_loss(
                                    (state2_hat_pred,
                                     state2_hat.detach()
                                     ), self.cfg.forward_loss, 'non_reduction'
                            ).sum(dim=1).unsqueeze(dim=1)

            _inverse_err = self.cfg.inverse_scale \
                            * self.get_loss(
                                    (pred_action, 
                                     actions.flatten().detach()
                                     ), self.cfg.inverse_loss
                            ).unsqueeze(dim=1)
            
            intrinsic_reward = self.cfg.eta * _forward_err
            rewards = intrinsic_reward.detach() # for
            if self.cfg.use_extrinsic:
                # print(rewards, rewards_.view(-1,1))
                rewards += rewards_.view(-1,1)
            rewards += self.cfg.gamma * torch.max(qvals_next)
            target_idx = torch.stack((torch.arange(actions.shape[0]), actions), dim=0)
            q_target = qvals.clone()
            q_target[target_idx.tolist()] = rewards.squeeze()
            _q_loss = self.cfg.qloss_scale \
                            * self.get_loss(
                                (F.normalize(qvals),
                                F.normalize(q_target.detach())
                                ), self.cfg.q_loss
                            )


            loss_ = self.cfg.beta * _forward_err \
                    + (1-self.cfg.beta) * _inverse_err
            # to test loss_ mean method
            # loss_ = torch.mean(loss_)
            loss_ = loss_.sum() / loss_.flatten().shape[0]
            loss = loss_ \
                    + self.cfg.lambda_ * _q_loss
            return loss

        
        elif loss_type == 'cross-entropy':
            y, y_hat = batches
            # print(batches)
            return self.cross_entropy(y, y_hat)
        
        elif loss_type == 'mse':
            y, y_hat = batches
            if args != (): 
                if args[0] == 'non_reduction':
                    return self.mse_non_reduction(y, y_hat)
            else:
                return self.mse(y, y_hat)
        

if __name__ == "__main__":
    cfg = dict(loss_fn='hihi')
    loss =LossFn(cfg)
    loss.get_name()
