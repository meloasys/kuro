import torch
from torch import nn
from torch.nn import functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_size, config):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(input_size,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin = nn.Linear(50,2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin = nn.Linear(25,1)
        
    def forward(self, x):
        x = F.normalize(x,dim=0)
        # forward common
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # forward actor
        x_actor = self.actor_lin(x)
        x_actor = F.log_softmax(x_actor, dim=1)
        # forward critic
        x_critic = F.relu(self.l3(x.detach()))
        x_critic = self.critic_lin(x_critic)
        x_critic = torch.tanh(x_critic)
        return x_actor, x_critic

if __name__ == "__main__":
    x = torch.randn((10,4))
    nn = ActorCritic()
    print(nn(x))

        
