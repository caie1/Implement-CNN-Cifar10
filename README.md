# Image Classification using CNN
This is a single-threaded python implemention of a Convolutional Neural network, with a pre-trained network included for classifying the CIFAR-10 dataset categories "Airplane", "Bird", and "Ship".

It was created as a final project for Indiana University's CSCI-B351 Course, Intro to Artificial Intelligence by Emma Cai and Aaron Leslie

Running interface.py will create a visual display of test images and their respective classifications by the network. These are images that the network was not trained on, but achieved 73% accuracy over after training.

![Interface](/images/interface.png)

main_network.py includes two functions, train() and validate() which respectively will train on a 3-category subset of CIFAR-10 training data, and will validate on a 3-category subset of CIFAR-10 test data.

Additionally, keras_comparison will run an effectively identical keras network over 40 epochs to compare the classifications of our network to. Keras achieves 3% more accuracy with identical settings, and 13% more with an identically sized network but better algorithms. Even better results could be achieved with a network with more ideal hyperparameters for this problem space.

![Keras Comparison](/images/keras_comparison.png)

All code is our own, unless cited (namely, snippets for loading CIFAR-10 data.) We used numpy for the array math for the CNN itself.