import numpy as np
import RL_Snakegame as env

done = False
while not done : 
    reward, state, done = env.env_step(np.array([1,0,0,0]), render=False)
    print(reward, state)