
Open source Deep Reinforcement Learning library  
 - refernece from    
    https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction   
    https://github.com/higgsfield/RL-Adventure (quantile regression dqn)   
    


Error
 - retrun args are different with nes_py 4 and gym 5   
    lib/python3.10/site-packages/nes_py/nes_env.py   
    315 add dict() to last return self.screen, reward, self.done, info, dict()
 - remove private attr   
    /home/melody/pyvenv/oasys/lib/python3.10/site-packages/gym/core.py   
    239 line

