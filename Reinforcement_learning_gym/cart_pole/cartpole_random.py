import gym

env = gym.make('CartPole-v0')
env.reset()

def random_games():
	for episode in range(6):
		env.reset()
		done = False
		score = 0
		for i in range(200):
			observation, reward, done, info = env.step(env.action_space.sample())

			if done == False :
				score += 1

			env.render()

		print(f'Trial {episode}')
		print('Score : ', score)
		print('-' * 25)
	env.close()

random_games()