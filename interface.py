import data_batch
from tkinter import Tk, Canvas, Frame, BOTH, Button
import time
import numpy as np
import CNN
import ANN
import random

class Network:

    def __init__(self):
        conv_layer_1 = CNN.CNNLayer(3, 3, 1, 32)

        conv_layer_2 = CNN.CNNLayer(32, 5, 2, 64)

        pooling_layer_1 = CNN.PoolingLayer(2,2)

        self.conv_layers = [conv_layer_1, conv_layer_2, pooling_layer_1]
        self.ann = ANN.ANN(2304, [128, 3])

        # 0 = Airplane
        # 2 = Bird
        # 8 = Ship

        conv_layer_1.load("data/3-network/CNNL1.npz")
        conv_layer_2.load("data/3-network/CNNL2.npz")
        self.ann.load("data/3-network/FCL.npz")
    
    def compute_category(self, data):
        data = self.conv_layers[0].forward(np.array([data]))
        for i in range(1, len(self.conv_layers)):
            data = self.conv_layers[i].forward(data)

        flattened = data.reshape(1, data.shape[1] * data.shape[2] * data.shape[3])

        output = self.ann.prop_forward(flattened)
        outputNum = -1
        biggest = -1
        for j in range(0, 3):
            if output[0][j] > biggest:
                outputNum = j
                biggest = output[0][j]
        return outputNum


class Display(Frame):

    def __init__(self):
        super().__init__()
        self.img = 0
        
        self.network = Network()

        self.batch = data_batch.DataBatch("cifar-10-python/cifar-10-batches-py/test_batch")
        filteredImages = []
        filteredLabels = []
        for x in range(len(self.batch.labels)):
            if(self.batch.labels[x] == 0):
                filteredImages.append(self.batch.images[x])
                filteredLabels.append(0)
            if(self.batch.labels[x] == 2):
                filteredImages.append(self.batch.images[x])
                filteredLabels.append(1)
            if(self.batch.labels[x] == 8):
                filteredImages.append(self.batch.images[x])
                filteredLabels.append(2)
        self.batch.images = np.array(filteredImages)
        self.batch.labels = np.array(filteredLabels)
        self.batch.label_names = np.array(["Airplane", "Bird", "Ship"])

        self.results = [("Classifying", x) for x in range(21)]

        self.canvas = Canvas()
        self.pack(fill=BOTH, expand=1)
        self.initUI()
    
    def increment(self):
        toCompute = 0
        for item, _ in self.results:
            if item is "Classifying":
                break
            toCompute += 1

        self.results[toCompute] = (self.network.compute_category(self.batch.images[self.img]), self.img)
        self.img += 1
        if(self.img >= 3000):
            self.img = 0

        self.initUI()

    def newImages(self):
        for x in range(21):
            new_index = self.img + x
            if(new_index >= 3000):
                new_index -= 3000
            self.results[x] = ("Classifying", new_index)
        self.b.destroy()
        self.initUI()

    def initUI(self):
        
        self.canvas.destroy()
        self.master.title("What if we... classified images? haha I'm just kidding ahaha... unless?")
        
        self.canvas = Canvas(self)

        img_size = 32
        scale = 7.5 # 7.5 for 1080p. 10 for 1440p
        right = 0
        seen = 0
        for y_img in range(3):
            for x_img in range(7):
                xOffset = x_img * scale * img_size * 1.1 + img_size * scale / 6
                yOffset = y_img * img_size * scale * 1.2 + img_size * scale / 10
                result, img_index = self.results[y_img * 7 + x_img]
                for x in range(32):
                    for y in range(32):
                        self.canvas.create_rectangle(xOffset + 5 + x*scale, yOffset + y*scale,
                        xOffset + 5 + x*scale + scale, yOffset + y*scale + scale,
                        outline='', fill="#%02x%02x%02x" % tuple((self.batch.images[img_index][y][x] * 255).astype(int)))
                self.canvas.create_text(xOffset + scale*img_size / 2, yOffset + scale * img_size + scale * img_size / 30, text="Actually " + self.batch.label_names[self.batch.labels[img_index]],
                font=("Comic Sans MS", int(scale * img_size / 18)))
                if result is "Classifying":
                    self.canvas.create_text(xOffset + scale*img_size / 2, yOffset + scale * img_size + scale * img_size / 8, text=result, 
                    fill="Black", font=("Comic Sans MS", int(scale * img_size / 18)))
                else:
                    self.canvas.create_text(xOffset + scale*img_size / 2, yOffset + scale * img_size + scale * img_size / 8, text="Network " + self.batch.label_names[result], 
                    fill=("green" if (result == self.batch.labels[img_index]) else "red"), font=("Comic Sans MS", int(scale * img_size / 18)))
                    if (result == self.batch.labels[img_index]):
                        right += 1
                    seen += 1
        if(seen > 0):
            self.canvas.create_text(xOffset/2 + img_size/2 * scale, yOffset + scale*img_size + scale * 10, text="Correct " + str(right) + "/" + str(seen)  
                + " - " + "{0:.0%}".format(right/seen), font=("Comic Sans MS", int(scale * img_size / 10)))
        self.canvas.pack(fill=BOTH, expand=1)
        
        unclassified = 0
        for item, _ in self.results:
            if item is "Classifying":
                unclassified += 1

        if unclassified > 0:
            self.after(50, self.increment)
        else:
            self.b = Button(self.master, text="New Images", command=self.newImages)
            self.b.place(x = xOffset/2 + img_size/2 * scale, y =  yOffset + scale*img_size + scale * 15)


def main():
    app = Display()
    app.mainloop()


if __name__ == '__main__':
    main()