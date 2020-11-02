import neuralnetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images=data['training_images']
    training_labels=data['training_labels']

layer_sizes=(784,128,128,10)

net = nn.NeuralNetwork(layer_sizes)
net.weights=np.load("weights1.npy",allow_pickle=True)
net.biases=np.load("biases1.npy",allow_pickle=True)
net.print_accuracy(training_images,training_labels)
