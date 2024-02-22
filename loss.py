import torch

class Loss:
    def __init__(self) -> None:
        self.name = 'loss_fn_parents'
    def get_name(self):
        print(self.name)

class LossFn(Loss):
    def __init__(self, config):
        super(LossFn,self).__init__()
        self.config = config
    def get_loss(self, batches, weight):

        if self.config['loss_fn'] == 'actor-critic':
            values, logprobs, returns = batches
            actor_loss = -1 * logprobs * (returns - values.detach())
            critic_loss = torch.pow(values - returns, 2)
            total_loss = (1-weight)*actor_loss.sum() + weight*critic_loss.sum()         
            return actor_loss, critic_loss, total_loss
        
        elif self.config['loss_fn'] == 'cross-entropy':
            '''works for distributed DQN'''
            x, y = batches
            loss = torch.Tensor([0.])
            loss.requires_grad = True
            for i in range(x.shape[0]):
                loss_ = -1 \
                        * torch.log(x[i].flatten(start_dim=0)) \
                        @ y[i].flatten(start_dim=0)
                loss += loss_

            return loss
        
        
        
    # def get_name(self):
    #     print(self.config)

if __name__ == "__main__":
    cfg = dict(loss_fn='hihi')
    loss =LossFn(cfg)
    loss.get_name()
