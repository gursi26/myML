import gym, os, sys, argparse
import numpy as np

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument('-f','--filename', help = 'enter path to q_table')
arg = parser.parse_args()

if arg.filename == None :
	q_table = np.load(current_path + '/Trained_qtables/q_table_24500.npy')
else :
	q_table = np.load(arg)


env = gym.make('MountainCar-v0')
min_vals = env.observation_space.low
max_vals = env.observation_space.high

num_buckets = [40] * len(max_vals)
bucket_size = (max_vals - min_vals) / num_buckets

def get_discrete_state(state):
	discrete_state = (state - min_vals) / bucket_size
	return tuple(discrete_state.astype(np.int))

current_state = env.reset()
current_discrete_state = get_discrete_state(current_state)

done = False 
while not done :

	action = q_table[current_discrete_state].argmax()
	new_state, reward, done, _ = env.step(action)

	current_discrete_state = get_discrete_state(new_state)
	env.render()

env.close()
