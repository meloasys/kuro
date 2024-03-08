import torch

def get_support(cfg):
    support = torch.linspace(cfg.dist_support_limit[0],
                            cfg.dist_support_limit[1],
                            cfg.support_div)
    return support


if __name__ == "__main__":
    pass