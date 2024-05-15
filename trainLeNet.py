from LeNet import Le_Net
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from datetime import datetime

train_data_x = np.load('balanced_gameplay_data.npy')
train_data_y = np.load("balanced_key_mappings.npy")



X_train, X_test, Y_train, Y_test = train_test_split(train_data_x, train_data_y, random_state=101, test_size=0.3)

for i,j in zip(train_data_x,train_data_y):
    print(i.shape)

WIDTH = 480
HEIGHT = 700
LR = 1e-3
EPOCHS = 10
input_shape = (WIDTH, HEIGHT, 3)
model = Le_Net(input_shape, 6)

try:
    history = model.fit(X_train, Y_train, epochs=EPOCHS, validation_data=(X_test, Y_test), verbose=1)
except KeyboardInterrupt:
    print("Stopping training due to keyboard interrupt")
    model.save('6out.h5')
else:
    # Save the trained model with a timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_filename = f'6out_10epochs{timestamp}.h5'
    model.save(model_filename)

