import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def keras_model(image_x, image_y):
    model = Sequential([
        Lambda(lambda x: x / 127.5 - 1., input_shape=(image_x, image_y, 1)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), padding='valid'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), padding='valid'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), padding='valid'),
        Flatten(),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss="mse")
    return model, [ModelCheckpoint("models/Autopilot_new.h5", verbose=1, save_best_only=True)]

def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels

def main():
    features, labels = loadFromPickle()
    features, labels = shuffle(features, labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0, test_size=0.3)
    train_x = train_x.reshape(train_x.shape[0], 100, 100, 1)
    test_x = test_x.reshape(test_x.shape[0], 100, 100, 1)
    model, callbacks_list = keras_model(100, 100)
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=10, batch_size=32, callbacks=callbacks_list)
    model.summary()

    model.save('models/Autopilot_new.h5')

if __name__ == "__main__":
    main()
    K.clear_session()
