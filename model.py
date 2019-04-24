import numpy as np
from activations import *

class neuralNet(object):

    def __init__(self, loss, optimizer, recurrentLoss=False):
        self.shape = [None, None]
        self.loss = loss
        self.optimizer = optimizer
        self.recurrentLoss = recurrentLoss
        self.layers = []
        
    def add(self, layer):
        currInput = layer.shape[1]
        if len(self.layers) > 0:
            prevOutput = self.layers[-1].shape[0]
            assert prevOutput == currInput
        else:
            self.shape[1] = currInput
        self.layers.append(layer)
        self.shape[0] = layer.shape[0]
        return None

    def fit(self, data, labels, epochs, verbose=True):
        self.optimizer.initializeState(self.layers)
        for epoch in range(epochs):
            x = data.copy()
            #Forward Pass:
            for layer in self.layers:
                x = layer.fwdPass(x)
            cost, dError = self.loss(x, labels, recurrent=self.recurrentLoss)
            #Backward Pass:
            gradients = []
            for layer in reversed(self.layers):
                dError = layer.bwdPass(dError)
                if isinstance(dError, tuple):
                    dError, grads = dError
                    gradients.append(grads)
                else:
                    gradients.append(None)
            #Param Update
            self.optimizer.update(self.layers, gradients)
            
            if verbose and epoch%10 == 0:
                print('Epoch: {0}; Cost: {1}'.format(epoch, cost))
        return None
    
    def predict(self, data):
        x = data.copy()
        #Forward Pass:
        for layer in self.layers:
            x = layer.fwdPass(x, inference=True)
        if self.loss == softmaxWithXentropy:
            x = softmax(x)
        return x
