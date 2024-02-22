import yaml, os
from pathlib import Path

class Config:
    def __init__(self):
        self.cfg_root = os.getenv('CFG_ROOT')
        cfg_name = os.getenv('CFG_NAME')
        print(self.cfg_root, cfg_name)
        dir = Path(self.cfg_root)/cfg_name
        with open(dir) as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        for i, k in enumerate(config.keys()):
            setattr(self, k, config[k])
    

if __name__ == '__main__':
    import dotenv, os, sys, torch
    dotenv.load_dotenv()
    cfg = Config()
    print(cfg.n_step_mode)
    