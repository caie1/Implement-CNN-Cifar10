import numpy as np
import matplotlib.pyplot as plt

""" Function credit to Alex Krizhevsky
https://www.cs.toronto.edu/~kriz/cifar.html """
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

""" Function credit to Magnus Erik Hvass Pedersen
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py """
def _convert_images(raw):
    channels = 3
    image_size = 32
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0
 
    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, channels, image_size, image_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


class DataBatch:
    def __init__(self, batch_file):
        self.raw_data = unpickle(batch_file)
        self.images = _convert_images(self.raw_data[b'data'])
        self.labels = self.raw_data[b'labels']
        self.label_names = unpickle("cifar-10-python/cifar-10-batches-py/batches.meta")[b'label_names']
