import torch
from torch import nn

class DistDQN(nn.Module):
    def __init__(self, input_size, *args, **kwargs) -> None:
        super(DistDQN, self).__init__(*args, **kwargs)
        self.l1 = nn.Linear(input_size, )


class DistDQN2:
    def __init__(self, params, action_space, cfg) -> None:
        # self.x = input
        self.theta = params
        self.action_space = action_space
        self.cfg = cfg 
        pass
    def forward(self, x):
        dim0, dim1, dim2, dim3 = self.cfg['dist_network_dim']
        t1 = dim0 * dim1
        t2 = dim1 * dim2
        theta1 = self.theta[0:t1].reshape(dim0, dim1)
        theta2 = self.theta[t1:t1+t2].reshape(dim1, dim2)
        l1 = self.x @ theta1
        l1 = torch.selu(l1)
        l2 = l1 @ theta2
        l2 = torch.selu(l2)
        l3 = []
        for i in range(self.action_space):
            step = dim2 * dim3
            theta5_dim = t1 + t2 + i*step
            theta5 = self.theta[theta5_dim:theta5_dim+step].reshape(dim2, dim3)
            l3_ = l2 @ theta5
            l3.append(l3_)
        l3 = torch.stack(l3, dim=1)
        l3 = torch.nn.functional.softmax(l3, dim=2)
        return l3.squeeze()

if __name__ == '__main__':
    nn = DistDQN2(1,1,1,1)
    print(nn)
