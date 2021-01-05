import gym, random, os, sys
import numpy as np
from statistics import mean, median
from collections import Counter

from keras import Input
from keras.models import Sequential
from keras.layers import Dropout, Dense

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))

env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500
# Number of frames considered to be a win
score_req = 85
# Score requirement for a random game to be used for training
initial_games = 40000
# These number of random games will be carried out


# Creating the initial population of training data for neural network
def initial_population():
	print('Generating training data....')
	x_vals, y_vals = [],[]
	scores = []
	accepted_scores = []

	for var in range(initial_games):
		score = 0
		game_memory = []
		previous_observation = []

		for var2 in range(goal_steps):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if len(previous_observation) > 0 :
				game_memory.append([list(previous_observation), action])

			previous_observation = observation
			score += reward

			if done :
				break

		if score >= score_req :
			accepted_scores.append(score)
			
			for data in game_memory : 	
				if data[1] == 1 :
					output = [0,1]
				elif data[1] == 0 :
					output = [1,0]

				x_vals.append(data[0]) 
				y_vals.append(output)

		env.reset()
		scores.append(score)

	x_vals = np.array(x_vals)
	y_vals = np.array(y_vals)

	np.save(file = current_path + '/model_stuff/x_vals_train.npy', arr = x_vals)
	np.save(file = current_path + '/model_stuff/y_vals_train.npy', arr = y_vals)

	print('Average accepted_scores : ', mean(accepted_scores))
	print('Median accepted_scores : ', median(accepted_scores))
	print('Training data generated.')
	print('-' * 75)

	return x_vals, y_vals


# Model for predicting the action based on the state
def neural_net(input_size):

	base_nodes = 512
	model = Sequential()

	model.add(Input(shape = (input_size,)))

	model.add(Dense(base_nodes, activation = 'relu'))
	model.add(Dropout(0.2))

	model.add(Dense(base_nodes * 2, activation = 'relu'))
	model.add(Dropout(0.2))

	model.add(Dense(base_nodes * 4, activation = 'relu'))
	model.add(Dropout(0.2))

	model.add(Dense(base_nodes * 2, activation = 'relu'))
	model.add(Dropout(0.2))

	model.add(Dense(base_nodes, activation = 'relu'))
	model.add(Dropout(0.2))

	model.add(Dense(2, activation = 'softmax'))

	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	print(model.summary())

	return model


# Training model on training data generated randomly
def train_model(x_vals, y_vals, model = False):

	if not model :
		model = neural_net(input_size = x_vals.shape[1])

	print('Training model...')
	model.fit(x_vals, y_vals, epochs = 3, verbose = 1)
	print('Training complete')
	return model


x_vals, y_vals = initial_population()
model = train_model(x_vals= x_vals, y_vals= y_vals)
model.save(current_path + '/model_stuff/cartpole_model4.h5')