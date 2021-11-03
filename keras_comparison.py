import numpy as np
import matplotlib.pyplot as plt
import data_batch
import random
import math
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPool2D
from keras.layers.core import Flatten, Dense, Activation
from keras.utils import np_utils
 
 
def main():
    """
    This class exists just for comparing results with Keras
    """
    batch_files = ["cifar-10-python/cifar-10-batches-py/data_batch_1",
    "cifar-10-python/cifar-10-batches-py/data_batch_2",
    "cifar-10-python/cifar-10-batches-py/data_batch_3",
    "cifar-10-python/cifar-10-batches-py/data_batch_4",
    "cifar-10-python/cifar-10-batches-py/data_batch_5"]

    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Convolution2D(64, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 0 = Airplane
    # 2 = Bird
    # 8 = Ship

    batch = data_batch.DataBatch(batch_files[0])
    filteredImages = []
    filteredLabels = []
    for filename in batch_files:
        batch = data_batch.DataBatch(filename)
        for x in range(len(batch.labels)):
            if(batch.labels[x] == 0):
                filteredImages.append(batch.images[x])
                filteredLabels.append([1,0,0])
            if(batch.labels[x] == 2):
                filteredImages.append(batch.images[x])
                filteredLabels.append([0,1,0])
            if(batch.labels[x] == 8):
                filteredImages.append(batch.images[x])
                filteredLabels.append([0,0,1])
    batch.images = np.array(filteredImages)
    batch.labels = np.array(filteredLabels)

    test = data_batch.DataBatch("cifar-10-python/cifar-10-batches-py/test_batch")
    filteredImages = []
    filteredLabels = []
    for x in range(len(test.labels)):
        if(test.labels[x] == 0):
            filteredImages.append(test.images[x])
            filteredLabels.append([1,0,0])
        if(test.labels[x] == 2):
            filteredImages.append(test.images[x])
            filteredLabels.append([0,1,0])
        if(test.labels[x] == 8):
            filteredImages.append(test.images[x])
            filteredLabels.append([0,0,1])
    test.images = np.array(filteredImages)
    test.labels = np.array(filteredLabels)

    model.fit(batch.images, batch.labels, batch_size=64, epochs=40, verbose=2, validation_data=(test.images, test.labels))

    input("done")

if __name__ == "__main__":
    main()
