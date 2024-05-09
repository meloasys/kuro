import torch, cv2
from skimage.transform import resize


###################### Dist-dqn
def get_support(cfg):
    support = torch.linspace(cfg.dist_support_limit[0],
                            cfg.dist_support_limit[1],
                            cfg.support_div)
    return support
###############################

########################### ICM 
def downscale_obs(obj, cfg):
    # obj = cv2.resize(obj, cfg.resize_scale, interpolation=cv2.INTER_LINEAR)
    # # obj = cv2.resize(obj1, (42,42), interpolation=cv2.INTER_LINEAR)
    # if cfg.gray_scale:    
    #     obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    # # print(obj)
    # print(obj)
    if cfg.gray_scale: 
        obj = resize(obj, cfg.resize_scale, anti_aliasing=True).max(axis=2)
    else:
        obj = resize(obj, cfg.resize_scale, anti_aliasing=True)
    # print(obj)
    return obj

def prepare_multi_state(state1, state2, cfg):
    state1 = state1.clone() # due to slice overwriting same memory space
    state2 = torch.from_numpy(
                downscale_obs(state2, cfg)
                ).float()
    for i in range(cfg.multi_states_size):
        state1[i] = state1[i+1] if i+1 != cfg.multi_states_size else state2
    return state1

def prepare_initial_state(state, cfg):
    state = torch.from_numpy(
                downscale_obs(state, cfg)
                ).float()
    tmp = state.repeat((cfg.multi_states_size,1,1))
    return tmp

def get_info(cfg, info):
    if 'SuperMarioBros' in cfg.env_name:
        last_pos = info['x_pos']
        return last_pos
    return

def eval_ep(cfg, info, additional_info):
    if 'SuperMarioBros' in cfg.env_name:
        if abs(info['x_pos'] - additional_info) < cfg.progress_thres:
            penalty = True
        else:
            penalty = False
            additional_info = info['x_pos']
        res = (penalty, additional_info)
    return res

#################### multihead-attention
def prepare_state(state, cfg):
    state = state.permute(2,0,1)
    max_v = state.flatten().max()
    state = state/max_v
    return state
def action_map(action, cfg):
    return torch.tensor([cfg.action_map[int(action)]])

if __name__ == "__main__":
    class cfg:
        def __init__(self) -> None:
            self.multi_states_size=5
            self.resize_scale=[5,5]
            self.gray_scale=True
    a = cv2.imread('run/t_ActorCritic_240322075115/test_res.jpg')
    b = cv2.imread('run/t_ActorCritic_240322075115/train_loss.jpg')
    a = prepare_initial_state(a, cfg())
    print(a.shape, a)
    a = prepare_multi_state(a, b, cfg())
    print(a.shape, a)
    pass