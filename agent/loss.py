import torch
from torch import nn

class Loss:
    def __init__(self) -> None:
        self.name = 'loss_fn_parents'
    def get_name(self):
        print(self.name)

class LossFn(Loss):
    def __init__(self, config):
        super(LossFn,self).__init__()
        self.cfg = config
        self.mse = nn.MSELoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')



    def get_loss(self, batches, loss_type):

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
        
        elif loss_type == 'cross-entropy':
            y, y_hat = batches
            return self.cross_entropy(y, y_hat)
        
        elif loss_type == 'mse':
            y, y_hat = batches
            return self.mse(y, y_hat)
        

if __name__ == "__main__":
    cfg = dict(loss_fn='hihi')
    loss =LossFn(cfg)
    loss.get_name()
