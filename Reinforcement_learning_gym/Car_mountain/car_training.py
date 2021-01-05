import gym
import numpy as np
import matplotlib.pyplot as plt
import os, sys

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))

env = gym.make('MountainCar-v0')

max_vals = env.observation_space.high
min_vals = env.observation_space.low
possible_actions = env.action_space.n

episodes = 25000
lr = 0.1
discount = 0.95

epsilon = 0.5
target_epsilon = 0.0001
epsilon_decay_value = (target_epsilon - epsilon) / (episodes/2)

num_buckets = [40] * len(max_vals)
bucket_size = (max_vals - min_vals) / num_buckets

q_table = np.random.uniform(low = -2, high = 0, size = (num_buckets + [possible_actions]))

def get_discrete_state(state):
	discrete_state = (state - min_vals) / bucket_size
	return tuple(discrete_state.astype(np.int))

show_every = 250
random_action_counter = 0
normal_action_counter = 0
episodes_madeit = 0

all_rewards = []
rewards_dict = {'avg': [], 'max': [], 'min': [], 'ep': []}

for single_episode in range(episodes):
	reward_per_episode = 0

	if single_episode % show_every == 0 :

		print('-' * 75)
		print("Attempt : ", single_episode)

		print('Normal actions : ', normal_action_counter)
		print('Random actions : ', random_action_counter)
		print(f'Success on {episodes_madeit} / {show_every} episodes')

		episodes_madeit = 0
		random_action_counter = 0
		normal_action_counter = 0
		render = True

	else :
		render = False

	discrete_state = get_discrete_state(env.reset())
	done = False
	while not done :

		if np.random.random() > epsilon :
			normal_action_counter += 1
			action = q_table[discrete_state].argmax()

		else :
			random_action_counter += 1
			action = np.random.randint(0, possible_actions)

		new_state, reward, done, _ = env.step(action)
		reward_per_episode += reward
		new_discrete_state = get_discrete_state(new_state)

		if render : 
			env.render()

		if not done :
			max_future_q = q_table[new_discrete_state].max()
			current_q = q_table[discrete_state + (action, )]

			new_q = (1 - lr) * current_q + lr * (reward + discount * max_future_q)

			q_table[discrete_state + (action, )] = new_q

		elif new_state[0] > env.goal_position : 
			episodes_madeit += 1
			q_table[discrete_state + (action, )] = 0

		discrete_state = new_discrete_state

	all_rewards.append(reward_per_episode)
	epsilon += epsilon_decay_value


	if single_episode % 500 == 0 :
		np.save(file = current_path + f'/Trained_qtables/q_table_{single_episode}', arr = q_table)

	if single_episode % show_every == 0 : 

		reward_values = all_rewards[-show_every:]
		average_reward = sum(reward_values)/len(reward_values)

		rewards_dict['ep'].append(single_episode)
		rewards_dict['avg'].append(average_reward)
		rewards_dict['max'].append(max(reward_values))
		rewards_dict['min'].append(min(reward_values))

		print(f'Episode : {single_episode} | Average : {average_reward} | Max : {max(reward_values)} | Min : {min(reward_values)}')


env.close()

x_vals = rewards_dict['ep']

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize = (11,5))

ax.plot(x_vals, rewards_dict['avg'], linewidth = 3, color = 'b', label = 'Average')
ax.plot(x_vals, rewards_dict['max'], linewidth = 3, color = 'green', label = 'Max')
ax.plot(x_vals, rewards_dict['min'], linewidth = 3, color = 'red', label = 'Min')

ax.set_title('Reward')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')

ax.grid(True)
ax.legend(loc = 'upper left')

fig.savefig(fname = current_path + '/reward_data.jpg', bbox_inches='tight')
plt.show()

np.save(file = current_path + f'/Trained_qtables/q_table_{single_episode}', arr = q_table)
