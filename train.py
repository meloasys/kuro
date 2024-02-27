import sys
sys.path.append('network')
sys.path.append('configs')

from utils import utils
from environment import env_manager
from network import nn_manager
from episode import episode, episode2
from agent import RLagent
from loss import LossFn
from config import Config
import torch.multiprocessing as mp

from pathlib import Path
import dotenv, os, sys, torch
dotenv.load_dotenv()

from network.actor_critic_nn import ActorCritic

# cfg = utils.get_configs()

# nn_network, network_names = nn_manager.get_network(cfg)
# if isinstance(nn_network, list):
#     for i, n in enumerate(network_names):
#         globals()[n] = nn_network[i]

# temp_env = env_manager.get_env(cfg)
# env_size = temp_env.reset()[0].shape[0]
# nn_cls_0 = globals()[network_names[0]](input_size=env_size) # single casse


def train_mp(proc_no, cfg, nn_cls_0, counter):
    env = env_manager.get_env(cfg)
    env.reset()

    optimizer = nn_manager.get_optimizer(
                                config=cfg, 
                                parameters=nn_cls_0.parameters(),
                                )
    optimizer.zero_grad()
    loss_fn = LossFn(cfg)

    episode_ = episode2(cfg)
    # create agent class
    agent_ = RLagent(
                nn_network=nn_cls_0,
                optimizer=optimizer,
                loss_inst=loss_fn,
                environment=env,
                episode=episode_,
                configs=cfg
                )

    # for i in range(cfg['epochs']):
    for i in range(cfg.epochs):
        # train agent class
        losses = agent_.run_update()
        if i % 200 == 0:
            agent_.test()        


if __name__ == '__main__':
    cfg = Config()
    print(vars(cfg))
    temp_env = env_manager.get_env(cfg)
    state_size = temp_env.reset()[0].shape[0]


    nn_network, network_names = nn_manager.get_network(cfg)
    if isinstance(nn_network, list):
        for i, n in enumerate(network_names):
            globals()[n] = nn_network[i]
    
    # single casse
    nn_cls_0 = globals()[network_names[0]](input_size=state_size,
                                           config=cfg) # single casse

    nn_cls_0.share_memory()
    procs = []
    epoch_counter_shared = mp.Value('i', 0)
    for i in range(cfg.num_multi_procs):
        proc = mp.Process(target=train_mp,
                          args=(i, cfg, nn_cls_0, epoch_counter_shared))
        proc.start()
        procs.append(proc)
    for p in procs:
        p.join()
    for p in procs:
        p.terminate()
