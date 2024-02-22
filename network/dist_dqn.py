import torch
from torch import nn
from torch.nn import functional as F

class DistDQN(nn.Module):
    def __init__(self, input_size, config, *args, **kwargs) -> None:
        super(DistDQN, self).__init__(*args, **kwargs)
        self.cfg = config
        self.l1 = nn.Linear(input_size, 
                            self.cfg.dist_network_dim[1])
        self.l2 = nn.Linear(self.cfg.dist_network_dim[1], 
                            self.cfg.dist_network_dim[2])
        self.l3 = torch.tensor([])
        for i in range(self.cfg.action_space):
            setattr(self, 
                    f'l3_{i}', 
                    nn.Linear(self.cfg.dist_network_dim[2], 
                              self.cfg.dist_network_dim[3]
                              )
                    )
            
    def forward(self, x):
        x = F.normalize(x, dim=0)
        x = torch.selu(self.l1(x))
        x = torch.selu(self.l2(x))
        for i in range(self.cfg.action_space):
            l3_tmp = getattr(self,f'l3_{i}')
            x_temp = l3_tmp(x).unsqueeze(dim=0)
            self.l3 = torch.cat([self.l3, x_temp], dim=0) 
        self.l3 = F.softmax(self.l3, dim=1)
        return self.l3
            


class DistDQN2:
    def __init__(self, params, action_space, cfg) -> None:
        # self.x = input
        self.theta = params
        self.action_space = action_space
        self.cfg = cfg 
        pass
    def forward(self, x):
        dims = self.cfg.dist_network_dim
        dim0, dim1, dim2, dim3 = dims
        t1 = dim0 * dim1
        t2 = dim1 * dim2
        theta1 = self.theta[0:t1].reshape(dim0, dim1)
        theta2 = self.theta[t1:t1+t2].reshape(dim1, dim2)
        l1 = x @ theta1
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
    import dotenv, os, sys, torch
    dotenv.load_dotenv()
    sys.path.append('./configs')
    from config import Config
    
    n_param = 128*100 + 100*25 + 25*51*3
    param = torch.randn(n_param)/10.
    model = DistDQN2(param,3,Config())
    res = model.forward(
                torch.randn((1,128))
                # torch.randn((128))
                )
    print(res.shape)

    model = DistDQN(128,Config())
    res = model(
        torch.randn((128))
             )
    print(res.shape)
