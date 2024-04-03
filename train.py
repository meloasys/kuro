import sys
sys.path.append('network')
sys.path.append('configs')
sys.path.append('agent')

from environment import env_manager
from network import nn_manager
from episode import episode
from agent import RLagent
from loss import LossFn
from config import Config
import torch.multiprocessing as mp

import dotenv, sys, datetime, os, warnings
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

dotenv.load_dotenv()
warnings.filterwarnings('ignore')

def train(proc_no, cfg, nn_cls_init, counter):
    env_manager.Environment.set_env(cfg)
    env = env_manager.Environment.get_env()
    env.reset()

    if len(cfg.network_models) == 1:
        optimizer = nn_manager.get_optimizer(
                                    config=cfg, 
                                    parameters=nn_cls_init.parameters(),
                                    )
    else:
        all_parameters = list()
        for n in nn_cls_init:
            all_parameters += list(nn_cls_init[n].parameters())
        optimizer = nn_manager.get_optimizer(
                                    config=cfg, 
                                    parameters=all_parameters,
                                    )
    
    optimizer.zero_grad()
    loss_fn = LossFn(cfg)

    episode_ = episode(cfg, env)
    # create agent class
    agent_ = RLagent(
                nn_network=nn_cls_init,
                optimizer=optimizer,
                loss_inst=loss_fn,
                environment=env,
                episode=episode_,
                configs=cfg
                )
    dt = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    # fname = 't_' + type(nn_cls_0).__name__ + '_' + dt
    fname = 't_' + cfg.nn_mod + '_' + dt
    os.makedirs('run/'+fname, exist_ok=True)
    losses = []
    for i in tqdm(range(cfg.epochs)):
    # for i in range(cfg.epochs):
        # train agent class
        loss = agent_.run_update()
        losses.append(loss)
        if i % cfg.test_interval == 0 and i > 1:
            # print(f'epoch {i} loss _ ', losses)
            plt.plot(np.arange(len(losses)), np.array(losses))
            plt.savefig(f'run/{fname}/train_loss.jpg')
            plt.clf()
            test_res = agent_.test()
            eplens_avg = test_res['eplens_avg']
            successes_avg = test_res['successes_avg']
            rewards_avg = test_res['rewards_avg']
            plt.plot(np.arange(len(eplens_avg)), np.array(eplens_avg))
            plt.plot(np.arange(len(successes_avg)), np.array(successes_avg))
            plt.plot(np.arange(len(rewards_avg)), np.array(rewards_avg))
            plt.savefig(f'run/{fname}/test_res.jpg')
            plt.clf()



if __name__ == '__main__':
    cfg = Config()
    print(vars(cfg))
    env_manager.Environment.set_env(cfg)
    temp_env = env_manager.Environment.get_env()
    state_size = temp_env.reset()[0].shape[0]


    nn_network, network_names = nn_manager.get_network(cfg)
    # initialize networks
    if isinstance(nn_network, list):
        nn_cls_init = dict()
        for i, n in enumerate(network_names):
            globals()[n] = nn_network[i](input_size=state_size,
                                        config=cfg)
            nn_cls_init[n] = globals()[n]


    # only support for single network
    if cfg.multi_procs and len(cfg.network_models) == 1:
        nn_cls_0 = nn_cls_init[network_names[0]]
        nn_cls_0.share_memory()
        procs = []
        epoch_counter_shared = mp.Value('i', 0)
        for i in range(cfg.num_multi_procs):
            proc = mp.Process(target=train,
                        args=(i, cfg, nn_cls_0, epoch_counter_shared))
            proc.start()
            procs.append(proc)
        for p in procs:
            p.join()
        for p in procs:
            p.terminate()
    else: # for single proc
        # for single network
        if len(cfg.network_models) == 1:
            train(0, cfg, nn_cls_init[network_names[0]], 0)
        # for multi network
        else:
            train(0, cfg, nn_cls_init, 0)



