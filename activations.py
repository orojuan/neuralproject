import numpy as np
   
def identity(x, deriv=False):
    if not deriv:
        return x
    return np.ones_like(x)

def sigmoid(x, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-x))
    return x * (1 - x)

def tanh(x, deriv=False):
    if not deriv:
        return np.tanh(x)
    return 1 - np.power(x, 2)

def relu(x, deriv=False):
    if not deriv:
        return np.maximum(0, x)
    return np.where(x <= 0, 0, 1)

def softmax(x, deriv=False):
    if not deriv:
        maxVal = np.max(x, axis=0, keepdims=True)
        x = x - maxVal
        return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
    # stack each prob vector as a diagonal matrix
    jacobians = np.einsum('ik,ij->kij', x, np.identity(x.shape[0]))
    # stack the outer product of each prob vector
    jacobians -= np.einsum('ik,jk->kij', x, x)
    return jacobians # [obs X dim X dim]