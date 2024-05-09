import torch, time


def get_network(cfg):
    mod_name = cfg.nn_mod
    networks = cfg.network_models
    instance_list = []
    for network in networks:
        tmp = __import__(mod_name)
        # print(tmp, network)
        globals()[network] = getattr(tmp, network)
        instance_list.append(globals()[network])
    return instance_list, networks
    
def get_optimizer(config, parameters):
    learning_rate = float(config.learning_rate)
    print('LEARNING RATE __________ ', learning_rate)
    time.sleep(1)
    if config.optimizer == 'adam':
        optim = torch.optim.Adam(lr=learning_rate,
                                 params=parameters)
    elif config.optimizer == 'SGD':
        optim = torch.optim.SGD(lr=learning_rate,
                                 params=parameters)
    assert optim is not None, 'Optimizer is not set'
    return optim
