from keras.models import Sequential
from keras.layers import Dense

import numpy as np


def extract_CSV():
    '''
    Recieve .csv file of Hololens coordinate data and
    transform it to numpy array for model training

    May need to be revised after receiving .csv data
    in order to properly reorder input data

    Return: xyz for position, rotation, magnitude
    '''

    file = open('data.csv', 'r')
    px, py, pz, rx, ry, rz, mx, my, mz = np.loadtxt(file, dtype='float32', delimiter=',', unpack=True)

    return (px, py, pz, rx, ry, rz, mx, my, mz)


print(extract_CSV())

x = [] # Inputs (9)
y = [] # Outputs (3)

# Create DNN Model
# 9 inputs (xyz's of position, rotation, magnitude)
model = Sequential()
model.add(Dense(9, input_dim=9)) # 1st layer (input layer)

# Insert additional hidden layers here

model.add(Dense(3)) # Bottom layer (output layer)

# Compile model:
# Loss Function: Mean squared error (difference between current and predict state)
# Optimizer: Adam (default, computationally efficient)
# Metrics: Mean Squared Error
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])

# Train model 
# Small run: 150 epochs and 10 batches
model.fit(x, y, batch_size=10, epochs=150)

