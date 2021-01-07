import numpy as np
import RL_snakegame as env

done = False
while not done : 
    reward, done = env.env_step(np.array([1,0,0,0]))