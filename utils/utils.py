import yaml, os
from pathlib import Path

def get_configs():
    cfg_root = os.getenv('CFG_ROOT')
    cfg_name = os.getenv('CFG_NAME')
    dir = Path(cfg_root)/cfg_name
    with open(dir) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    get_configs('./configs/base.yaml')