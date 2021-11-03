import numpy as np
import ANN
import data_batch
import random
import math
import CNN

def trainOn(data, labels, layer_list, ann, verbose, learning_rate, momentum):
    """
    Train given layers over data and labels with given parameters

    Arguments:
    data -- numpy array of images with shape (image_count, height, width, color_channels)
    labels -- int labels for images with shape (image_count)
    layer_list -- CNN and Max pooling layer list
    ANN -- Fully connected network for output
    verbose -- Boolean value for function verbosity
    learning_rate -- factor to adjust network weights and biases by
    momentum -- momentum percentage for network weight and bias updating
    """
    for layer in layer_list:
        data = layer.forward(data)

    flattened = data.reshape(data.shape[0], data.shape[1] * data.shape[2] * data.shape[3])

    output = ann.prop_forward(flattened)
    desired_output = np.zeros((data.shape[0], 3))

    right = 0
    for i in range(0, data.shape[0]):
        desired_output[i][labels[i]] = 1
        outputNum = 0
        biggest = -1000
        for j in range(0, 3):
            if output[i][j] > biggest:
                outputNum = j
                biggest = output[i][j]
        if outputNum == labels[i]:
            right += 1
    if(verbose):
         print("Right: " + str(right) + " / " + str(data.shape[0]) + " - " + "{0:.0%}".format(right/data.shape[0]))
    
    deltas = ann.prop_backward(desired_output, learning_rate, momentum)

    deltas = deltas.reshape(deltas.shape[0], data.shape[1], data.shape[2], data.shape[3])

    for layer in reversed(layer_list):
        deltas = layer.backward(deltas, learning_rate, momentum)

    return layer_list,  ann

                   
def train():
    batch_files = ["cifar-10-python/cifar-10-batches-py/data_batch_1",
    "cifar-10-python/cifar-10-batches-py/data_batch_2",
    "cifar-10-python/cifar-10-batches-py/data_batch_3",
    "cifar-10-python/cifar-10-batches-py/data_batch_4",
    "cifar-10-python/cifar-10-batches-py/data_batch_5"]
    conv_layer_1 = CNN.CNNLayer(3, 3, 1, 32)

    conv_layer_2 = CNN.CNNLayer(32, 5, 2, 64)

    pooling_layer_1 = CNN.PoolingLayer(2,2)

    conv_layers = [conv_layer_1, conv_layer_2, pooling_layer_1]
    ann = ANN.ANN(2304, [128, 3])

    # 0 = Airplane
    # 2 = Bird
    # 8 = Ship

    conv_layer_1.load("data/3-network/CNNL1.npz")
    conv_layer_2.load("data/3-network/CNNL2.npz")
    ann.load("data/3-network/FCL.npz")
    
    for epoch in range(0, 40):
        print ("epoch " + str(epoch))
        batch_sizes = 64
        for filename in batch_files:
            batch = data_batch.DataBatch(filename)
            filteredImages = []
            filteredLabels = []
            for x in range(len(batch.labels)):
                if(batch.labels[x] == 0):
                    filteredImages.append(batch.images[x])
                    filteredLabels.append(0)
                if(batch.labels[x] == 2):
                    filteredImages.append(batch.images[x])
                    filteredLabels.append(1)
                if(batch.labels[x] == 8):
                    filteredImages.append(batch.images[x])
                    filteredLabels.append(2)
            batch.images = np.array(filteredImages)
            batch.labels = np.array(filteredLabels)

            print("Running on file " + filename[-12:])
            order = list(range(0, len(batch.images), batch_sizes))
            random.shuffle(order)
            for i in order:
                conv_layers, ann = trainOn(batch.images[i:i+batch_sizes], batch.labels[i:i+batch_sizes],
                conv_layers, ann, True, 0.000004 * math.pow(0.96, epoch), 0.7) #Decay and momentum
            print("saving")
            conv_layer_1.save("data/3-network/CNNL1.npz")
            conv_layer_2.save("data/3-network/CNNL2.npz")
            ann.save("data/3-network/FCL.npz")
    
def validate():
    conv_layer_1 = CNN.CNNLayer(3, 3, 1, 32)

    conv_layer_2 = CNN.CNNLayer(32, 5, 2, 64)

    pooling_layer_1 = CNN.PoolingLayer(2,2)

    conv_layers = [conv_layer_1, conv_layer_2, pooling_layer_1]
    ann = ANN.ANN(2304, [128, 3])

    # 0 = Airplane
    # 2 = Bird
    # 8 = Ship

    conv_layer_1.load("data/3-network/CNNL1.npz")
    conv_layer_2.load("data/3-network/CNNL2.npz")
    ann.load("data/3-network/FCL.npz")
    
    batch = data_batch.DataBatch("cifar-10-python/cifar-10-batches-py/test_batch")
    filteredImages = []
    filteredLabels = []
    for x in range(len(batch.labels)):
        if(batch.labels[x] == 0):
            filteredImages.append(batch.images[x])
            filteredLabels.append(0)
        if(batch.labels[x] == 2):
            filteredImages.append(batch.images[x])
            filteredLabels.append(1)
        if(batch.labels[x] == 8):
            filteredImages.append(batch.images[x])
            filteredLabels.append(2)
    batch.images = np.array(filteredImages)
    batch.labels = np.array(filteredLabels)
    data = batch.images
    for i in range(len(conv_layers)):
        data = conv_layers[i].forward(data)

    flattened = data.reshape(data.shape[0], data.shape[1] * data.shape[2] * data.shape[3])

    output = ann.prop_forward(flattened)

    right = 0
    seen = 0
    for i in range(batch.images.shape[0]):
        outputNum = -1
        biggest = -1
        for j in range(0, 3):
            if output[i][j] > biggest:
                outputNum = j
                biggest = output[i][j]
        seen+=1
        if outputNum == batch.labels[i]:
            right += 1
    print("Right: " + str(right) + " / " + str(seen) + " - " + "{0:.0%}".format(right/seen))


def main():
   # train()
    validate()
    
if __name__ == "__main__":
    main()
