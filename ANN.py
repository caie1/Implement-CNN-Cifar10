import numpy as np
import matplotlib.pyplot as plt
import data_batch
import random

class ANN:
    
    def __init__(self, inputs, layer_sizes):
        """
        Initialize artificial neural network

        Arguments:
        inputs -- the number of inputs to our network
        layer_sizes -- an array of numbers representing neurons in each layer, desired quantity of outputs should equal the last layer
        """
        self.layer_count = len(layer_sizes)

        self.weights = []
        self.biases = []

        self.weights_momentum = []
        self.biases_momentum = []

        np.random.seed(1) #Seed our random number for testing consistency
        for i in range(self.layer_count):
            layer_weights = np.random.randn(inputs, layer_sizes[i]) / 5 #Our array contains the weights for each input of each neuron
            layer_biases = np.ones(layer_sizes[i])

            self.weights.append(layer_weights)
            self.weights_momentum.append(np.zeros_like(layer_weights))
            self.biases.append(layer_biases)
            self.biases_momentum.append(np.zeros_like(layer_biases))

            inputs = layer_sizes[i] #Our inputs into the next layer is the size of our last layer

    
    def prop_forward(self, input_layer):
        """
        This is the forward propagation for our ANN, handling all of our layers at once
        
        Arguments:
        input_layer -- array of inputs for ANN with shape (image_number, inputs)
        
        Returns:
        Z -- output of forward propagation with shape (image_number, outputs)
        """
        (image_number, _) = input_layer.shape
        # Loop through the batch of training examples
        Z = []
        self.cache = []
        for i in range(image_number):
            self.cache.append([])
            inputs = input_layer[i]
            self.cache[i].append(inputs) #Cache our results for each layer for back-prop
           
            for l in range(len(self.weights)): #Number of Layers
                if l == len(self.weights) - 1:
                    inputs = self.softMax(np.dot(inputs, self.weights[l]) + self.biases[l])
                    inputs = inputs.reshape(inputs.shape[0])
                else:
                    inputs = self.LeakyReLU(np.dot(inputs, self.weights[l]) + self.biases[l])
                self.cache[i].append(inputs) #Cache our results for each layer for back-prop

            Z.append(inputs)
        self.cache = np.array(self.cache)
        return Z

    def prop_backward(self, desired_result, learning_rate, momentum):
        """
        This is the backward propagation for our ANN, assuming we just called forward prop on the input that we want to give desired_result

        Arguments:
        desired_result -- array of desired values of our last prop_forward inputs in shape (image_number, values)
        learning_rate -- rate at which weights and biases update with respect to error
        momentum -- percentage of carry-over of previous deltas into current calculation

        Returns:
        deltas -- an array of the deltas of the input layer of shape (image_number, inputs)
        """
        (image_number, _) = desired_result.shape
        weight_change = np.zeros_like(self.weights)
        bias_change = np.zeros_like(self.biases)
        deltas = np.zeros((image_number, len(self.weights[0])))
        for i in range(image_number):
            delta = 0
            for l in reversed(range(self.layer_count)):
                if l == self.layer_count - 1:
                    delta = -desired_result[i] + self.cache[i][l + 1]
                    delta = np.square(delta) * np.sign(delta)
                    weight_change[l] += (delta * self.cache[i][l].reshape((-1, 1)))
                    bias_change[l] += delta
                else:
                    error = np.dot(self.weights[l+1], delta)
                    delta = error * self.d_LeakyReLU(self.cache[i][l+1])
                    weight_change[l] += (delta * self.cache[i][l].reshape((-1, 1)))
                    bias_change[l] += delta
            deltas[i] = np.dot(self.weights[0], delta)  * self.d_LeakyReLU(self.cache[i][0])
        for l in reversed(range(self.layer_count)):
            self.weights_momentum[l] = (self.weights_momentum[l] * momentum) + (weight_change[l] * learning_rate)
            self.biases_momentum[l] = (self.biases_momentum[l] * momentum) + (bias_change[l] * learning_rate)
            self.weights[l] -= self.weights_momentum[l]
            self.biases[l] -= self.biases_momentum[l]
        return deltas


    def save(self, fileName):
        """
        Saves entire ANN
        """
        np.savez_compressed(fileName, a=self.weights, b=self.biases)

    def load(self, fileName):
        """
        Loads entire ANN
        """
        data = np.load(fileName, allow_pickle=True)
        self.weights = data['a']
        self.biases = data['b']

    def softMax(self, Z):
        """
        Softmax activation function
        """
        exp = np.exp(Z - Z.max())
        return exp/np.sum(exp)
        
    def LeakyReLU(self, Z):
        """
        Leaky ReLU activation funtion
        """
        return np.where(Z > 0, Z, Z*0.01)
    
    
    def d_LeakyReLU(self, Z):
        """
        The derivatative of Leaky ReLU used for backward propogation
        """
        return np.where(Z > 0, 1, 0.01)
