Open source Deep Reinforcement Learning library

Modification on Deep Reinforcement Learning In Action
https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction

Error
- retrun args are different with nes_py 4 and gym 5
lib/python3.10/site-packages/nes_py/nes_env.py
315 add dict() to last return self.screen, reward, self.done, info, dict()