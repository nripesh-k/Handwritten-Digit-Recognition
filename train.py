import neuralnetwork as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images=data['training_images']
    training_labels=data['training_labels']

layer_sizes=(784,32,10)

net = nn.NeuralNetwork(layer_sizes)
# net.weights=np.load("weights1.npy",allow_pickle=True)
# net.biases=np.load("biases1.npy",allow_pickle=True)
# net.print_accuracy(training_images,training_labels)

k=0
initial=k
scale=40000
for i in range(10):
    k=initial
    while (k-initial)<scale:  
        for img,lbl in zip(training_images[k:k+500],training_labels[k:k+500]):
            net.train(img,lbl)
        #print("Learning Progress: {0}%".format((k-initial)/(scale)*100))
        k=k+500 
    print("Learning Progress: {0}%".format((i)/(10)*100))

net.print_accuracy(training_images[40000:],training_labels[40000:])

net.print_accuracy(training_images,training_labels)

q=input("Do you want to save the network:(y/n)")
if q=="y":
    np.save("w_32.npy",net.weights)
    np.save("b_32.npy",net.biases)