import yaml, os
from pathlib import Path

class Config:
    def __init__(self):
        self.cfg_root = os.getenv('CFG_ROOT')
        cfg_name = os.getenv('CFG_NAME')
        print('###################### Config root and name:', 
                                        self.cfg_root, cfg_name)
        dir = Path(self.cfg_root)/cfg_name
        with open(dir) as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        for i, k in enumerate(config.keys()):
            setattr(self, k, config[k])
        default_cfg_name = os.getenv('DEFAULT_CFG')
        dir = Path(self.cfg_root)/default_cfg_name
        with open(dir) as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        for i, k in enumerate(config.keys()):
            if self.nn_mod in config[k]:
                setattr(self, k, True)
            else:
                setattr(self, k, False)
        if self.value_nn:
            if hasattr(self, 'prob_q'):
                pass
            else:
                raise ValueError('prob_q should be included \
                                in config file for value_nn')
    
    def dump2yaml(self, file_to) -> None:
        file_to = file_to / 'config.yaml'
        with open(file_to, 'w+') as f:
            yaml.dump(vars(self), f, allow_unicode=True)
        

if __name__ == '__main__':
    import dotenv, os
    dotenv.load_dotenv()
    cfg = Config()
    print(cfg.batch_size)

    cfg.batch_size = 10000

    cfg.dump2yaml()
    