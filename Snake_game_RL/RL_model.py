import numpy as np
import RL_Snakegame as env

render = True
game_updater = env.render_or_not(render = render)

done = False

while not done : 
    action = np.random.randint(0,3)
    reward, state, done = env.env_step(action = action, render = render, game_update_speed = game_updater)
    print(reward, state)