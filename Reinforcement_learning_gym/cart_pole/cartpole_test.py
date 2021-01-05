from keras.models import load_model
import gym, os, sys
import numpy as np

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))
env = gym.make('CartPole-v1')
model = load_model(current_path + '/model_stuff/cartpole_model3.h5')
scores = []
games = 10

for trial in range(games):
	score = 0
	done = False
	model_input = np.array(env.reset(), ndmin = 2)

	while not done :
		action_pred = model.predict(model_input).argmax()
		new_state, reward, done, _ = env.step(action_pred)
		env.render()
		model_input = np.array(new_state, ndmin = 2)
		score += reward

	scores.append(score)
	print(f'Trial {trial + 1} score : {score}')

env.close()
avg_score = sum(scores)/len(scores)

print('-' * 50)
print('Average score : ', avg_score)