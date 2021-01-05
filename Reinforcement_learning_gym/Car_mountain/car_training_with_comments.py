import gym
import numpy as np
from IPython.display import clear_output

env = gym.make('MountainCar-v0')

max_vals = env.observation_space.high
min_vals = env.observation_space.low
possible_actions = env.action_space.n

print('Highest value for states : ', max_vals)
print('Lowest value for states : ', min_vals)
print('Number of possible actions : ', possible_actions)
print()

# Highest value for states :  [0.6  0.07]
# Lowest value for states :  [-1.2  -0.07]
# Number of possible actions :  3

episodes = 25000
lr = 0.1
discount = 0.95
# discount compares importance of future reward to current reward

epsilon = 0.5
target_epsilon = 0.1

epsilon_decay_value = (target_epsilon - epsilon) / episodes
print(epsilon_decay_value)
# Epsilon refers to how often you want the agent to do a random exploratory action
# This allows the agent to find alternative ways to complete the same objective

num_buckets = [20] * len(max_vals)
bucket_size = (max_vals - min_vals) / num_buckets
print('Range within each bucket : ', bucket_size)
print()

q_table = np.random.uniform(low = -2, high = 0, size = (num_buckets + [possible_actions]))
print('Shape of table : ', q_table.shape)
print()
# creates a table with random q values for every combinattion of 2 states
# shape of table is [20,20,3]

def get_discrete_state(state):
	discrete_state = (state - min_vals) / bucket_size
	return tuple(discrete_state.astype(np.int))

show_every = 500
for single_episode in range(episodes):

	if single_episode % show_every == 0 :
		print("Attempt : ", single_episode)
		render = True
	else :
		render = False

	discrete_state = get_discrete_state(env.reset())
	done = False
	while not done :

		if np.random.random() > epsilon :
			#print('Normal action')
			action = q_table[discrete_state].argmax()

		else :
			#print('Random action ----')
			action = np.random.randint(0, possible_actions)
		# action = 0 : push car left
		# action = 1 : do nothing
		# action = 2 : push car right

		new_state, reward, done, _ = env.step(action)
		# State is the way the environment responds to our actions
		# In this case, state returns position and velocity of car
		# If done is returned as true, the loop breaks

		new_discrete_state = get_discrete_state(new_state)

		if render : 
			env.render()

		if not done :
			max_future_q = q_table[new_discrete_state].max()
			current_q = q_table[discrete_state + (action, )]

			new_q = (1 - lr) * current_q + lr * (reward + discount * max_future_q)
			# formula for new_q

			q_table[discrete_state + (action, )] = new_q

		elif new_state[0] > env.goal_position : 
			print(f"Made it on episode {single_episode}")
			q_table[discrete_state + (action, )] = 0

		discrete_state = new_discrete_state

	epsilon -= epsilon_decay_value


env.close()

#np.save('/Users/gursi/Desktop/ML/Reinforcement_Learning/final_Qtable', q_table)
print('Done')

