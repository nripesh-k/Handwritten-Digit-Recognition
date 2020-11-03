import numpy as np

class NeuralNetwork:
    
    def __init__(self,layer_sizes):
        self.lr=0.0005
        weight_shapes=[(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
        self.weights=[np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
        self.biases=[np.zeros((s,1)) for s in layer_sizes[1:]]
        self.active=[np.zeros(s) for s in layer_sizes[1:]]
        self.error=[np.zeros(s) for s in layer_sizes[1:]]
    
    def predict(self,a):
        i=0
        for w,b in zip(self.weights,self.biases):
            a=self.activation(np.matmul(w,a)+b)
            self.active[i]=a
            i=i+1
        return a
    
    def print_accuracy(self,images,labels):
        predictions=self.predict(images)
        num_correct=sum([np.argmax(a)==np.argmax(b) for a,b in zip(predictions,labels)])
        print('{0}/{1} accuracy: {2}%'.format(num_correct,len(images),(num_correct/len(images))*100))
        
    def train(self,image,label):
        self.predict(image)
        i=-1
        self.error[i]=label-self.active[i]
        for e,w in zip(list(reversed(self.error[:-1])),list(reversed(self.weights[1:]))):
            weight=w
            e=np.matmul(weight.transpose(),self.error[i])
            i=i-1
            self.error[i]=e
        #correction
        for i in range(len(self.error)-1):
            self.error[i]*=self.active[i]*self.lr
            self.biases[i]+=self.error[i]
        
        self.weights[0]+=np.matmul(self.error[0],image.transpose())
        j=1
        for w,e,a in zip(self.weights[1:],self.error[1:],self.active[:-1]):
            self.weights[j]+=np.matmul(e,a.transpose())
            j=j+1


        # for e,z in zip(self.error,self.active):
        #     for eele,aele in zip(e,z):

    # def train(self,label):

    @staticmethod
    def activation(x):
        return 1/(1+np.exp(-x))