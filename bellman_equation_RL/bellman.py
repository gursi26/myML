import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pandas as pd

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))
image_path = current_path + '/maze.jpg'
image = plt.imread(image_path)

# Showing the maze
plt.figure(figsize = (7,5))
plt.imshow(image)
plt.axis(False)

# Loading in the reward matrix
reward_matrix = pd.read_csv(current_path + '/reward.csv')
cols = np.array(reward_matrix.columns)
reward = np.array(reward_matrix)

# Dictionary to map maze node letters to numbers
letter_to_num_dict = {}
num_to_letter_dict = {}
for var1 in range(9):
    letter_to_num_dict[cols[var1]] = var1
    num_to_letter_dict[var1] = cols[var1]

# Initial empty qtable
qtable = np.zeros((9,9))

# Finds possible actions given state
def possible_actions(position):
    current_row = reward[position,]
    possible_move = np.where(current_row > 0)[0]
    return possible_move

# Selects on of the possible actions
def make_move(position):
    possible_moves = possible_actions(position)
    move = np.random.choice(possible_moves)
    return move

# Calculates new q value base on state and action and updates table
def calc_qval(current_state, action, lr):
    current_row = qtable[action,]
    max_value = current_row.max()
    qtable[current_state, action] = reward[current_state, action] + (lr * max_value)

# Main training loop
start_state = np.random.randint(0,7)
epochs = 50000
learning_rate = 0.8
for epoch in range(epochs):
    action = make_move(start_state)
    calc_qval(start_state, action, learning_rate)
    start_state = action

# Normalizes the output table, giving values as percentages
def normalize(qtable):
    maximum_value = qtable.max()
    qtable = (qtable/maximum_value) * 100
    return qtable

final_table = normalize(qtable)
final_df = pd.DataFrame(final_table, columns=cols)
final_df.head(9)

# Maps moves using finished table give starting position
def map_moves(start_state):
    while start_state != 8 :
        action = final_table[start_state].argmax()
        print(f'{num_to_letter_dict[start_state]} --> {num_to_letter_dict[action]}')
        start_state = action

for i in range(8):
    print(f'Attempt {i + 1}')
    start_state = i
    print('Start position : ', num_to_letter_dict[start_state])
    map_moves(start_state)
    print('-'*30)

plt.show()