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
        for i in range(self.cfg.action_space):
            setattr(self, 
                    f'l3_{i}', 
                    nn.Linear(self.cfg.dist_network_dim[2], 
                              self.cfg.dist_network_dim[3]
                              )
                    )
            
    def forward(self, x):
        self.l3 = []
        x = F.normalize(x, dim=1)
        x = torch.selu(self.l1(x))
        x = torch.selu(self.l2(x))
        for i in range(self.cfg.action_space):
            l3_tmp = getattr(self,f'l3_{i}')
            x_temp = l3_tmp(x)
            self.l3.append(x_temp)
        self.l3 = torch.stack(self.l3, dim=1)
        self.l3 = F.softmax(self.l3, dim=2)
        return self.l3
            



if __name__ == '__main__':
    import dotenv, sys, torch
    dotenv.load_dotenv()
    sys.path.append('./configs')
    from config import Config
    
    n_param = 128*100 + 100*25 + 25*51*3
    param = torch.randn(n_param)/10.
    model = DistDQN(param,3,Config())
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
