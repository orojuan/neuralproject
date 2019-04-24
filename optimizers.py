import numpy as np

class sgd(object):
    def __init__(self, learnRate, momentum=0.0):
        self.learnRate = learnRate
        self.momentum = momentum
        self.velocity = []

    def initializeState(self, layers):
        for layer in reversed(layers):
            if layer.trainable:
                init = {key: np.zeros_like(val) for key, val in layer.params.items()}
            else:
                init = None
            self.velocity.append(init)
        return None
    
    def update(self, layers, gradients):
        momentum = self.momentum
        for i, layer in enumerate(reversed(layers)):
            if layer.trainable:
                if self.momentum > 0.0:
                    for key in gradients[i].keys():
                        self.velocity[i][key] = momentum*self.velocity[i][key] + (1-momentum)*gradients[i][key]
                    change = self.velocity[i].copy()
                else:
                    change = gradients[i]
                layer.paramUpdate(change, self.learnRate)
        return None

class rmsProp(object):
    def __init__(self, learnRate, momentum=0.9):
        self.learnRate = learnRate
        self.momentum = momentum
        self.velocity = []

    def initializeState(self, layers):
        for layer in reversed(layers):
            if layer.trainable:
                init = {key: np.zeros_like(val) for key, val in layer.params.items()}
            else:
                init = None
            self.velocity.append(init)
        return None
    
    def update(self, layers, gradients):
        momentum = self.momentum
        for i, layer in enumerate(reversed(layers)):
            if layer.trainable:
                for key in gradients[i].keys():
                    self.velocity[i][key] = momentum*self.velocity[i][key] + (1-momentum)*np.power(gradients[i][key], 2)
                change = gradients[i].copy()
                for key in change.keys():
                    change[key] = change[key] / (np.sqrt(self.velocity[i][key]) + 1e-8)
                layer.paramUpdate(change, self.learnRate)
        return None

class adam(object):
    def __init__(self, learnRate, beta1=0.9, beta2=0.999):
        self.learnRate = learnRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.velocity1 = []
        self.velocity2 = []
        self.iteration = 1

    def initializeState(self, layers):
        for layer in reversed(layers):
            if layer.trainable:
                init1 = {key: np.zeros_like(val) for key, val in layer.params.items()}
                init2 = init1.copy()
            else:
                init1 = None
                init2 = None
            self.velocity1.append(init1)
            self.velocity2.append(init2)
        return None

    def update(self, layers, gradients):
        beta1, beta2 = self.beta1, self.beta2
        for i, layer in enumerate(reversed(layers)):
            if layer.trainable:
                for key in gradients[i].keys():
                    self.velocity1[i][key] = beta1*self.velocity1[i][key] + (1-beta1)*gradients[i][key]
                    self.velocity2[i][key] = beta2*self.velocity2[i][key] + (1-beta2)*np.power(gradients[i][key], 2)
                corrected1 = self.velocity1[i].copy()
                corrected2 = self.velocity2[i].copy()
                for key in corrected1.keys():
                    corrected1[key] = corrected1[key] / (1 - np.power(beta1, self.iteration))
                    corrected2[key] = corrected2[key] / (1 - np.power(beta2, self.iteration))
                change = corrected1.copy()
                for key in change.keys():
                    change[key] = change[key] / (np.sqrt(corrected2[key]) + 1e-8)
                layer.paramUpdate(change, self.learnRate)
        self.iteration += 1
        return None