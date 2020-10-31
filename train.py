import neuralnetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images=data['training_images']
    training_labels=data['training_labels']

layer_sizes=(784,128,128,10)

net = nn.NeuralNetwork(layer_sizes)
net.weights=np.load("weights.npy",allow_pickle=True)
net.biases=np.load("biases.npy",allow_pickle=True)
net.print_accuracy(training_images[:500],training_labels[:500])

k=20000
initial=k
scale=10000
while (k-initial)<scale:  
    for i in range(10):
        for img,lbl in zip(training_images[k:k+500],training_labels[k:k+500]):
            net.train(img,lbl)
    print("Progress: {0}%".format((k-initial)/(scale)*100))
    k=k+500

net.print_accuracy(training_images,training_labels)
np.save("weights.npy",net.weights)
np.save("biases.npy",net.biases)