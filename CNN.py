import numpy as np
import ANN
import data_batch
import random
import math
 
class CNNLayer:
      
    def __init__(self, input_channels, filter_size, stride, filter_count):
        """
        Initialize a CNN Layer
        
        Arguments:
        input_channels -- a number for how many input channels this layer will receive
        filter-size -- a number representing the length and height of the filter
        stride -- a number representing how far the filter should move in each step
        filter_counter -- a number representing the amount of filters to be used in this step
        """
        self.f = filter_size
        self.stride = stride
        self.input_layer = None
        self.input_layer_padded = None

        #Random initial filters centered around 0 with small variation
        self.weights = np.random.randn(self.f, self.f, input_channels, filter_count)
        #Initialize biases to be small non-zero positive number (0.1)
        self.biases = np.ones((1, 1, 1, filter_count))

        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)
        
            
    def unit_conv_step_with_ReLU(self, input_slice, W, b):
        """
        Apply one filter (W) to a single slice (input_slice)
        
        Arguments:
        input_slice -- a slice input data of shape (f, f, input_channel) where f is the length of filter
        W -- the weight matrix of shape (f, f, input_channel) which is the filter itself
        b -- the bias matrix of shape (1, 1, 1)
        
        Return:
        Z -- a single value which is the result of a single piece of convolution
        """
        volume = input_slice * W
        Z = np.sum(volume)
        Z = Z + float(b)
        Z = self.ReLU(Z)
        return Z
    
     
    def forward(self, input_layer):
        """
        This is the forward propagation go through the convolution layer
        
        Arguments:
        input_layer -- the previous layer of shape (image_number, prev_height, prev_width, prev_channel)
        
        Returns:
        Z -- the output of the convolution layer and active layer with the shape (image_number, new_height, new_width, new_channel)
             new_height = (prev_height - f + 2*pad) // stride + 1
             new_width = (prev_width - f + 2*pad) // stride + 1
             new_channel = the number of filters ( == filter_number)
        """
        # Retrieve data and info we already know
        self.input_layer = input_layer
        (image_number, prev_height, prev_width, _) = input_layer.shape
        (f, _, _, filter_number) = self.weights.shape
        
        # Compute the dimensions of the output
        new_height = (prev_height - f) // self.stride + 1
        new_width = (prev_width - f) // self.stride + 1
        
        # Initialize the output Z with zeros
        Z = np.zeros((image_number, new_height, new_width, filter_number))
        
        
        # Loop through the batch of training examples
        for i in range(image_number):
            
            # Select the ith padded training example
            piece_input_layer = self.input_layer[i, :, :, :]
            
            # Loop through vertical axis of output image
            for h in range(new_height):
                # Loop through horizontal axis of output image
                for w in range(new_width):
                    # Loop through the channels of output image
                    for c in range(filter_number):
                        vert_start = h * self.stride
                        vert_end = vert_start + f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + f
                        input_slice = piece_input_layer[vert_start : vert_end, horiz_start : horiz_end, :]
                        Z[i, h, w, c] = self.unit_conv_step_with_ReLU(input_slice, self.weights[:, :, :, c], self.biases[:, :, :, c])
        
        return Z
      
    def ReLU(self, Z):
        """
        Leaky ReLU activate funtion
        """
        return np.where(Z > 0, Z, Z*0.01)
     
    def d_ReLU(self, Z):
        """
        The derivatative of Leaky ReLU used for backward propogation
        """
        return np.where(Z > 0, 1, 0.01)
              
    def backward(self, dZ, learning_rate, momentum):
        """
        Backward Propogation for relu and convolution layer 
        
        Argument:
        dZ -- the gradient get from later layer of shape (image_number, new_height, new_width, new_channel)
              which is the same of output of conv_forward_with_ReLU
        learning_rate -- rate at which weights and biases update with respect to error
        momentum -- percentage of carry-over of previous deltas into current calculation
              
        Returns:
        dx -- the gradient pass to former layer of shape (image_number, prev_height, prev_width, prev_channel)
              which is the same of input_layer which is the input of pool_forward
        """
        (image_number, new_height, new_width, filter_number) = dZ.shape

        dx = np.zeros(self.input_layer.shape)
        dW = np.zeros(self.weights.shape)
        db = np.zeros(self.biases.shape)
        
        for i in range(image_number):
            input_slice = self.input_layer[i,: ,:, :]
            for h in range(new_height):
                for w in range(new_width):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f
                        i_slice = input_slice[vert_start : vert_end, horiz_start : horiz_end, : ]
                        for c in range(filter_number):
                            dx[i, vert_start : vert_end, horiz_start : horiz_end, : ] += dZ[i, h, w, c] * self.weights[:, :, :, c]
                            dW[:, :, :, c] +=  dZ[i, h, w, c] * i_slice
                            db[:, :, :, c] += dZ[i, h, w, c]

        self.weight_momentum = (self.weight_momentum * momentum) + (dW * learning_rate)
        self.bias_momentum = (self.bias_momentum * momentum) + (db * learning_rate)

        self.weights -= self.weight_momentum
        self.biases -= self.bias_momentum

        return dx * self.d_ReLU(self.input_layer)

    def save(self, fileName):
        """
        Saves CNN Layer
        """
        np.savez_compressed(fileName, a=self.weights, b=self.biases)

    def load(self, fileName):
        """
        Loads CNN Layer
        """
        data = np.load(fileName, allow_pickle=True)
        self.weights = data['a']
        self.biases = data['b']

        
        
             
class PoolingLayer:
     
    def __init__(self, filter_size, stride):
        """
        Initialize a Pooling Layer
        
        Arguments:
        filter_size -- a number representing the size of the filter N*N
        stride -- a number representing how far the filter should move in each step
        """
        self.f = filter_size
        self.stride = stride
        self.input_layer = None

    def forward(self, input_layer):
        """
        forward propagation for max pooling
        
        Arguments:
        input_layer -- the previous layer of shape (image_number, prev_height, prev_width, channel)
        
        Return:
        Z -- the output of the max pooling layer with the shape (image_number, new_height, new_width, channel)
             new_height = (prev_height - self.f)//self.stride + 1
             new_width = (prev_width - self.f)//self.stride + 1
             
        """
        self.input_layer = input_layer
        (image_number, prev_height, prev_width, channel) = input_layer.shape
        new_height = (prev_height - self.f)//self.stride + 1
        new_width = (prev_width - self.f)//self.stride + 1
        Z = np.zeros((image_number, new_height, new_width, channel))
        
        for i in range(image_number):
            for h in range(new_height):
                for w in range(new_width):
                    for c in range(channel):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f
                        input_slice = input_layer[i, vert_start : vert_end, horiz_start : horiz_end, c]
                        Z[i, h, w, c] = np.max(input_slice)

        return Z

    def backward(self, dZ, _, __):
        """
        backward propogation of pooling layer
        
        Argument:
        dZ -- the gradient get from later layer of shape (image_number, new_height, new_width, channel)
              which is the same of output of pool_forward
        
        Return:
        dx -- the gradient pass to fromer layer of shape (image_number, prev_height, prev_width, channel)
              which is the same of input_layer which is the input of pool_forward
        """
        (image_number, new_height, new_width, channel) = dZ.shape
        dx = np.zeros((self.input_layer.shape))
        
        for i in range(image_number):
            input_slice = self.input_layer[i,: ,: ,:]
            for h in range(new_height):
                for w in range(new_width):
                    for c in range(channel):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f
                        cut = input_slice[vert_start : vert_end, horiz_start : horiz_end, c]
                        biggest = np.argmax(cut)
                        dx[i, vert_start+(int(biggest / self.f)), horiz_start+(biggest % self.f), c] += dZ[i, w, h, c]
        return dx
     