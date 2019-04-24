import numpy as np
from scipy.stats import ortho_group
from abc import ABCMeta, abstractmethod
from activations import *

def initializer(weightShape, fanIn, method):
    if method == 'xavier':
        C = 1
    elif method == 'he':
        C = 2
    return C * np.random.randn(*weightShape) / np.sqrt(np.prod(fanIn))

class neuralNetLayer(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, shape):
        self.shape = shape
        self.trainable = True
    
    @abstractmethod
    def fwdPass(self, x):
        pass
    
    @abstractmethod
    def bwdPass(self, d_out):
        pass
    
    @abstractmethod
    def paramUpdate(self, paramsChange, learnRate):
        for param, change in paramsChange.items():
            self.params[param] -= learnRate*change
        return None

class activation(neuralNetLayer):
    def __init__(self, shape, activation):
        neuralNetLayer.__init__(self, shape)
        self.activation = activation
        self.trainable = False
        self.cache = []
    
    def fwdPass(self, x, inference=False):
        self.cache = []
        a = self.activation(x)
        self.cache.append(a)
        return a
    
    def bwdPass(self, dOut):
        a = self.cache[0]
        if self.activation.__name__ == 'softmax':
            jacobians = self.activation(a, deriv=True)
            # stack vectors columnwise that result of multiplying
            # the jacobian of each observation with its error signal
            dIn = np.einsum('kij,jk->ik', jacobians, dOut)
        else:
            dIn =  dOut*self.activation(a, deriv=True)
        return dIn
    
    def paramUpdate(self):
        raise AttributeError( "'Activation' layer has no method 'paramUpdate'." )

class dense(neuralNetLayer):
    def __init__(self, shape, activation, W=None, b=None, initMethod='xavier'):
        neuralNetLayer.__init__(self, shape)
        if W is None:
            W = initializer(shape, shape[1], initMethod)
        if b is None:
            b = np.zeros((shape[0], 1))
        self.activation = activation
        self.params = {'W':W, 'b':b}
        self.cache = []
    
    def fwdPass(self, x, inference=False):
        self.cache = []
        W, b = self.params['W'], self.params['b']
        z = np.matmul(W, x) + b
        a = self.activation(z)
        self.cache.append(x)
        self.cache.append(z)
        self.cache.append(a)
        return a
    
    def bwdPass(self, d_out, cache=None):
        if cache is not None:
            x, z, a = cache
        else:
            x, z, a = self.cache
        W = self.params['W']
        grads = {}
        if self.activation.__name__ == 'softmax':
            jacobians = self.activation(a, deriv=True)
            # stack vectors columnwise that result of multiplying
            # the jacobian of each observation with its error signal
            dz = np.einsum('kij,jk->ik', jacobians, d_out)
        else:
            dz = d_out*self.activation(a, deriv=True) # [o X m]
        grads['W'] = np.matmul(dz, x.T) # [o X m] X [m X i] = [o X i]
        grads['b'] = np.sum(dz, axis=1, keepdims=True)
        dIn = np.matmul(W.T, dz) # [i X o] X [o X m] = [i X m]
        return dIn, grads

#NOTE: inverted dropout is used!!
class dropout(neuralNetLayer):
    def __init__(self, shape, dropRate=0.5):
        neuralNetLayer.__init__(self, shape)
        self.dropRate = dropRate
        self.trainable = False
        self.cache = []
    
    def fwdPass(self, x, inference=False):
        if inference:
            return x
        self.cache = []
        mask = np.random.binomial(1, 1.0-self.dropRate, x.shape)
        a = x * mask / (1.0-self.dropRate)
        self.cache.append(mask)
        return a
    
    def bwdPass(self, dOut):
        mask = self.cache[0]
        dIn = dOut * mask / (1.0-self.dropRate) # [i X m]
        return dIn
    
    def paramUpdate(self):
        raise AttributeError( "'Dropout' layer has no method 'paramUpdate'." )

class batchNorm(neuralNetLayer):
    
    eps = 1e-5

    def __init__(self, shape, gamma=None, beta=None, momentum=0.99):
        neuralNetLayer.__init__(self, shape)
        numDim = shape[-1] if isinstance(shape[-1], int) else shape[-1][-1]
        if gamma is None:
            gamma = np.ones((numDim, 1))
        if beta is None:
            beta = np.zeros((numDim, 1))
        self.params = {'gamma':gamma, 'beta':beta}
        self.momentum = momentum
        self.movingMeans = np.zeros((numDim, 1))
        self.movingVars = np.ones((numDim, 1))
        self.cache = []
        
    def fwdPass(self, x, inference=False):
        if inference:
            if len(x.shape) == 2:
                means, variances = self.movingMeans, self.movingVars
            elif len(x.shape) == 4:
                means = self.movingMeans.reshape(1,1,1,-1)
                variances = self.movingVars.reshape(1,1,1,-1)
            return (x - means) / np.sqrt(variances + self.eps)
        gamma, beta = self.params['gamma'], self.params['beta']
        numDim = gamma.shape[0]
        if len(x.shape) == 2:
            obs = x.shape[1]
            axis = 1
        elif len(x.shape) == 4:
            obs = np.sum(x.shape[0:3])
            axis = (0,1,2)
            gamma = gamma.reshape(1,1,1,numDim)
            beta = beta.reshape(1,1,1,numDim)
        self.cache = []
        
        avgs = 1/float(obs) * np.sum(x, axis=axis, keepdims=True) # [d X 1]
        self.movingMeans = self.momentum*self.movingMeans + (1-self.momentum)*avgs.reshape(numDim, 1)
        self.cache.append(avgs)
        
        diffs = x - avgs # [d X m]
        self.cache.append(diffs)
        
        sq_diffs = np.power(diffs, 2) # [d X m]
        self.cache.append(sq_diffs)
        
        variances = 1/float(obs) * np.sum(sq_diffs, axis=axis, keepdims=True) # [d X 1]
        self.movingVars = self.momentum*self.movingVars + (1-self.momentum)*variances.reshape(numDim, 1)
        self.cache.append(variances)
        
        stddevs = np.sqrt(variances + self.eps) # [d X 1]
        self.cache.append(stddevs)
        
        inv_stddevs = 1/stddevs # [d X 1]
        self.cache.append(inv_stddevs)
        
        std_x = inv_stddevs * diffs # [d X m]
        self.cache.append(std_x)
        
        scaled_x = gamma * std_x # [d X m]
        self.cache.append(scaled_x)
        
        shifted_x = scaled_x + beta # [d X m]
        
        return shifted_x
    
    def bwdPass(self, dOut):
        gamma = self.params['gamma']
        numDim = gamma.shape[0]
        if len(dOut.shape) == 2:
            obs = dOut.shape[1]
            axis = 1
        elif len(dOut.shape) == 4:
            obs = np.sum(dOut.shape[0:3])
            axis = (0,1,2)
            gamma = gamma.reshape(1,1,1,numDim)
        
        avgs, diffs, sq_diffs, variances, stddevs = self.cache[:5]
        inv_stddevs, std_x, scaled_x = self.cache[5:]
        grads = {}

        # Step 9
        d_scaled_x = dOut # [d X m]
        grads['beta'] = np.sum(dOut, axis=axis, keepdims=True).reshape(numDim, 1) # [d X 1]
        # Step 8
        d_std_x = gamma * d_scaled_x # [d X m]
        grads['gamma'] = np.sum(np.multiply(std_x, d_scaled_x), axis=axis, keepdims=True).reshape(numDim, 1) # [d X 1]
        # Step 7
        d_diffs = inv_stddevs * d_std_x # [d X m]
        d_inv_stddevs = np.sum(diffs*d_std_x, axis=axis, keepdims=True) # [d X 1]
        # Step 6
        d_stddevs = -1/np.power(stddevs, 2) * d_inv_stddevs # [d X 1]
        # Step 5
        d_variances = 0.5 * np.power(variances + self.eps, -0.5) * d_stddevs # [d X 1]
        # Step 4
        d_sq_diffs = 1/float(obs) * np.ones(sq_diffs.shape) * d_variances # [d X m]
        # Step 3
        d_diffs += 2 * diffs * d_sq_diffs # [d X m]
        # Step 2
        dIn = d_diffs # [d X m]
        d_avgs = -np.sum(d_diffs, axis=axis, keepdims=True) # [d X 1]
        # Step 1
        dIn += 1/float(obs) * np.ones(scaled_x.shape) * d_avgs # [d X m]
        return dIn, grads

class rnn(neuralNetLayer):
    def __init__(self, shape, steps, Wsx=None, Wss=None, bs=None, initMethod='xavier', watchState=False, returnSeq=False):
        neuralNetLayer.__init__(self, shape)
        if Wsx is None:
            Wsx = initializer(shape, shape[1], initMethod)
        if Wss is None:
            Wss = ortho_group.rvs(shape[0])
        if bs is None:
            bs = np.zeros((shape[0], 1))
        self.steps = steps
        self.watchState = watchState
        self.returnSeq = returnSeq
        self.params = {'Wsx':Wsx, 'Wss':Wss, 'bs':bs}
        self.cache = []
    
    def fwdPass(self, X, state0=None, inference=False):
        def fwdStep(x, prevState):
            Wsx, Wss, bs = self.params['Wsx'], self.params['Wss'], self.params['bs']
            state = tanh(np.matmul(Wss, prevState) + np.matmul(Wsx, x) + bs)
            self.cache.append((state, prevState, x))
            return state
        
        dim_s, dim_x = self.shape 
        obs = X.shape[1]
        self.cache = []
        state = np.zeros((dim_s, obs, self.steps))
        if state0 is None:
            state0 = np.zeros((dim_s, obs))
        state_t = state0
        for t in range(self.steps):
            state_t = fwdStep(X[:,:,t], state_t)
            state[:,:,t] = state_t
        if not self.returnSeq:
            state = state[:,:,-1]
        return state
    
    def bwdPass(self, dOut):
        def bwdStep(dNextState, cache):
            state, prevState, x = cache
            Wsx, Wss = self.params['Wsx'], self.params['Wss']
            
            dtanh = tanh(state, deriv=True)*dNextState
            
            dx = np.matmul(Wsx.T, dtanh) # [dim_x X dim_h] X [dim_h X m] = [dim_x X m]
            dWsx = np.matmul(dtanh, x.T) # [dim_h X m] X [m X dim_x] = [dim_h X dim_x]
            
            dPrevState = np.matmul(Wss.T, dtanh) # [dim_h X dim_h] X [dim_h X m] = [dim_h X m]
            dWss = np.matmul(dtanh, prevState.T) # [dim_h X m] X [m X dim_h] = [dim_h X dim_h]
            dbs = np.sum(dtanh, axis=1, keepdims=True) # [dim_h X 1]
            return dPrevState, dx, dWsx, dWss, dbs
        
        dim_s, dim_x = self.shape 
        obs = self.cache[0][2].shape[1]
        if not self.returnSeq:
            dOut = np.concatenate((np.zeros((dim_s, obs, self.steps-1)), dOut.reshape(dim_s, obs, 1)), axis=2)
        grads = {}
        dIn = np.zeros((dim_x, obs, self.steps))
        grads['Wsx'] = np.zeros((dim_s, dim_x))
        grads['Wss'] = np.zeros((dim_s, dim_s))
        grads['bs'] = np.zeros((dim_s, 1))
        dPrevState_t = np.zeros((dim_s, obs))
        
        for t in reversed(range(self.steps)):
            dPrevState_t, dx_t, dWsx_t, dWss_t, dbs_t = bwdStep(dPrevState_t + dOut[:,:,t], self.cache[t])
            dIn[:,:,t] += dx_t
            grads['Wsx'] += dWsx_t
            grads['Wss'] += dWss_t
            grads['bs'] += dbs_t
        dState0 = dPrevState_t
        if self.watchState:
            return dState0, dIn, grads
        return dIn, grads

class lstm(neuralNetLayer):
    def __init__(self, shape, steps, Wf=None, bf=None, Wi=None, bi=None, Wc=None, bc=None, Wo=None, bo=None, initMethod='xavier', watchState=False, returnSeq=False):
        neuralNetLayer.__init__(self, shape)
        if Wf is None:
            Wf = ortho_group.rvs(shape[0])
            Wf = np.concatenate((Wf, initializer(shape, shape[1], initMethod)), axis=1)
        if bf is None:
            bf = np.zeros((shape[0], 1))
        if Wi is None:
            Wi = ortho_group.rvs(shape[0])
            Wi = np.concatenate((Wi, initializer(shape, shape[1], initMethod)), axis=1)
        if bi is None:
            bi = np.zeros((shape[0], 1))
        if Wc is None:
            Wc = ortho_group.rvs(shape[0])
            Wc = np.concatenate((Wc, initializer(shape, shape[1], initMethod)), axis=1)
        if bc is None:
            bc = np.zeros((shape[0], 1))
        if Wo is None:
            Wo = ortho_group.rvs(shape[0])
            Wo = np.concatenate((Wo, initializer(shape, shape[1], initMethod)), axis=1)
        if bo is None:
            bo = np.zeros((shape[0], 1))
        self.steps = steps
        self.watchState = watchState
        self.returnSeq = returnSeq
        self.params = {'Wf':Wf,'bf':bf,'Wi':Wi,'bi':bi,'Wc':Wc,'bc':bc,'Wo':Wo,'bo':bo}
        self.cache = []
    
    def fwdPass(self, X, state0=None, inference=False):
        def fwdStep(x, prevState, prevMemory):
            dim_s, dim_x = self.shape 
            obs = x.shape[1]
            Wf, Wi, Wc, Wo = self.params['Wf'], self.params['Wi'], self.params['Wc'], self.params['Wo']
            bf, bi, bc, bo = self.params['bf'], self.params['bi'], self.params['bc'], self.params['bo']
            concat = np.zeros((dim_s + dim_x, obs))
            concat[: dim_s, :] = prevState
            concat[dim_s :, :] = x
            
            forget = sigmoid(np.matmul(Wf, concat) + bf)
            update = sigmoid(np.matmul(Wi, concat) + bi)
            dummy = tanh(np.matmul(Wc, concat) + bc)
            memory = forget*prevMemory + update*dummy
            expose = sigmoid(np.matmul(Wo, concat) + bo)
            state = expose*tanh(memory)
            
            self.cache.append((state, memory, prevState, prevMemory, forget, update, dummy, expose, x))
            return state, memory
        
        dim_s, dim_x = self.shape 
        obs = X.shape[1]
        self.cache = []
        state = np.zeros((dim_s, obs, self.steps))
        memory = np.zeros((dim_s, obs, self.steps))
        #it does not make sense to have non-zero visible state when hidden state is zero!!
        memory_t = np.zeros((dim_s, obs))
        if state0 is None:
            state0 = np.zeros((dim_s, obs))
        state_t = state0
        
        for t in range(self.steps):
            state_t, memory_t = fwdStep(X[:,:,t], state_t, memory_t)
            state[:,:,t] = state_t
            memory[:,:,t] = memory_t 
        if not self.returnSeq:
            state = state[:,:,-1]
        if self.watchState:
            return state, memory
        return state
    
    def bwdPass(self, dOut):
        def bwdStep(dNextState, dNextMemory, cache):
            state, memory, prevState, prevMemory, forget, update, dummy, expose, x = cache
            dim_s, dim_x = self.shape
            _, obs = dNextState.shape
            Wf, Wi, Wc, Wo = self.params['Wf'], self.params['Wi'], self.params['Wc'], self.params['Wo']
            
            do = dNextState*tanh(memory)
            dc_prime = dNextState*expose
            dfc = dNextMemory + dc_prime*(1-np.power(tanh(memory), 2))
            dPrevMemory = dfc*forget
            dig = dfc.copy()
            df = dfc*prevMemory
            di = dig*dummy
            dg = dig*update
            daf = df*forget*(1-forget)
            dai = di*update*(1-update)
            dag = dg*(1-np.power(dummy, 2))
            dao = do*expose*(1-expose)
            da = np.concatenate((daf, dai, dag, dao)).reshape(-1, obs)
            Wx = np.concatenate((Wf[:,dim_s:],Wi[:,dim_s:],Wc[:,dim_s:],Wo[:,dim_s:]))
            Wh = np.concatenate((Wf[:,:dim_s],Wi[:,:dim_s],Wc[:,:dim_s],Wo[:,:dim_s]))
            dx = np.matmul(Wx.T, da)
            dPrevState = np.matmul(Wh.T, da)
            dWx = np.matmul(da, x.T)
            dWh = np.matmul(da, prevState.T)
            db = da
            
            dWf = np.concatenate((dWh[:dim_s,:], dWx[:dim_s,:]), axis=1)
            dbf = np.sum(db[:dim_s,:], axis=1, keepdims=True)
            dWi = np.concatenate((dWh[dim_s:2*dim_s,:], dWx[dim_s:2*dim_s,:]), axis=1)
            dbi = np.sum(db[dim_s:2*dim_s,:], axis=1, keepdims=True)
            dWc = np.concatenate((dWh[2*dim_s:3*dim_s,:], dWx[2*dim_s:3*dim_s,:]), axis=1)
            dbc = np.sum(db[2*dim_s:3*dim_s,:], axis=1, keepdims=True)
            dWo = np.concatenate((dWh[3*dim_s:,:], dWx[3*dim_s:,:]), axis=1)
            dbo = np.sum(db[3*dim_s:,:], axis=1, keepdims=True)
            
            return dPrevState, dPrevMemory, dx, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo
        
        dim_s, dim_x = self.shape 
        obs = self.cache[0][8].shape[1]
        if not self.returnSeq:
            dOut = np.concatenate((np.zeros((dim_s, obs, self.steps-1)), dOut.reshape(dim_s, obs, 1)), axis=2)
        grads = {}
        dIn = np.zeros((dim_x, obs, self.steps))
        dPrevState_t = np.zeros((dim_s, obs))
        dPrevMemory_t = np.zeros((dim_s, obs))
        grads['Wf'] = np.zeros((dim_s, dim_s + dim_x))
        grads['Wi'] = np.zeros((dim_s, dim_s + dim_x))
        grads['Wc'] = np.zeros((dim_s, dim_s + dim_x))
        grads['Wo'] = np.zeros((dim_s, dim_s + dim_x))
        grads['bf'] = np.zeros((dim_s, 1))
        grads['bi'] = np.zeros((dim_s, 1))
        grads['bc'] = np.zeros((dim_s, 1))
        grads['bo'] = np.zeros((dim_s, 1))
        
        for t in reversed(range(self.steps)):
            dPrevState_t, dPrevMemory_t, dx_t, dWf_t, dbf_t, dWi_t, dbi_t, dWc_t, dbc_t, dWo_t, dbo_t = bwdStep(dPrevState_t + dOut[:,:,t], dPrevMemory_t, self.cache[t])
            dIn[:,:,t] += dx_t
            grads['Wf'] += dWf_t
            grads['Wi'] += dWi_t
            grads['Wc'] += dWc_t
            grads['Wo'] += dWo_t
            grads['bf'] += dbf_t
            grads['bi'] += dbi_t
            grads['bc'] += dbc_t
            grads['bo'] += dbo_t
            
        dState0 = dPrevState_t
        if self.watchState:
            return dState0, dIn, grads
        return dIn, grads

class denseThruTime(neuralNetLayer):
    def __init__(self, shape, steps, activation):
        neuralNetLayer.__init__(self, shape)
        self.layer = dense(shape, activation)
        self.params = self.layer.params
        self.steps = steps
        self.cache = []
    
    def fwdPass(self, X, inference=False):
        self.cache = []
        dim_x, obs, steps = X.shape
        A = np.zeros((self.shape[0], obs, steps))
        for t in range(steps):
            A[:,:,t] = self.layer.fwdPass(X[:,:,t])
            self.cache.append(self.layer.cache)
        return A
    
    def bwdPass(self, dOut):
        hidden, obs, steps = dOut.shape
        dIn = np.zeros((self.shape[1], obs, steps))
        grads = {}
        grads['W'] = np.zeros((self.shape[0], self.shape[1]))
        grads['b'] = np.zeros((self.shape[0], 1))
        for t in reversed(range(steps)):
            dIn_t, denseGrads = self.layer.bwdPass(dOut[:,:,t], self.cache[t])
            dIn[:,:,t] += dIn_t
            grads['W'] += denseGrads['W']
            grads['b'] += denseGrads['b']
        return dIn, grads
    
    def paramUpdate(self, paramsChange, learnRate):
        for param, change in paramsChange.items():
            #self.layer.params[param] -= learnRate*change
            self.params[param] -= learnRate*change
        self.layer.params = self.params
        return None

class repeatVector(neuralNetLayer):
    def __init__(self, shape, steps):
        neuralNetLayer.__init__(self, shape)
        self.steps = steps
        self.trainable = False
    
    def fwdPass(self, x, inference=False):
        echo = np.repeat(x[:, :, np.newaxis], self.steps, axis=2)
        return echo
    
    def bwdPass(self, dOut):
        dIn = np.sum(dOut, axis=2)
        return dIn
    
    def paramUpdate(self):
        raise AttributeError( "'RepeatVector' layer has no method 'paramUpdate'." )

class conv2D(neuralNetLayer):
    def __init__(self, inputShape, filters, filterSize, stride, pad, dataFormat, W=None, b=None, initMethod='xavier'):
        assert dataFormat == 'channels_last'
        height = int(1 + (inputShape[0] - filterSize + 2*pad) / stride)
        width = int(1 + (inputShape[1] - filterSize + 2*pad) / stride)
        channels = inputShape[-1]
        shape = ((height,width,filters), inputShape)
        neuralNetLayer.__init__(self, shape)
        
        self.filters = filters
        self.filterSize = filterSize
        self.stride = stride
        self.pad = pad
        if W is None:
            W = initializer((filterSize, filterSize, channels, filters), filterSize*filterSize*channels, initMethod)
        if b is None:
            b = np.zeros((1,1,1,filters))
        self.params = {'W':W, 'b':b}
        self.cache = []
    
    def fwdPass(self, X, inference=False):
        def convolution(x_slice, W_filter, bias):
            a = np.sum(x_slice * W_filter) + float(bias)
            return a
        
        self.cache = []
        (obs, H_prev, W_prev, C_prev) = X.shape
        W, b = self.params['W'], self.params['b']
        Height, Width = self.shape[0][:2]
        A = np.zeros((obs, Height, Width, self.filters))
        pad = self.pad
        X_pad = np.pad(X, ((0,0),(pad, pad),(pad, pad),(0,0)), 'constant')
        
        for i in range(obs):
            for h in range(Height):
                for w in range(Width):
                    for c in range(self.filters):
                        # Find the corners of the current "slice"
                        vert_start = h * self.stride
                        vert_end = vert_start + self.filterSize
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.filterSize
                        # Use the corners to define the current slice on the ith training example of X, channel c
                        x_slice = X_pad[i,vert_start:vert_end,horiz_start:horiz_end,:]
                        A[i, h, w, c] = convolution(x_slice, W[:,:,:,c], b[0,0,0,c])
        self.cache.append(X)
        return A
    
    def bwdPass(self, d_Out):
        X = self.cache[0]
        (obs, H_prev, W_prev, C_prev) = X.shape
        (obs, Height, Width, C) = d_Out.shape
        W = self.params['W']
        grads = {}
        dIn = np.zeros((obs, H_prev, W_prev, C_prev))                           
        grads['W'] = np.zeros((self.filterSize, self.filterSize, C_prev, C))
        grads['b'] = np.zeros((1, 1, 1, C))
        pad = self.pad
        X_pad = np.pad(X, ((0,0),(pad, pad),(pad, pad),(0,0)), 'constant')
        d_In_pad = np.pad(dIn, ((0,0),(pad, pad),(pad, pad),(0,0)), 'constant')
        
        for i in range(obs):                       # loop over the training examples
            # select ith training example from A_prev_pad and dA_prev_pad
            x_pad = X_pad[i,:,:,:]
            d_in_pad = d_In_pad[i,:,:,:]
            for h in range(Height):                   # loop over vertical axis of the output volume
                for w in range(Width):               # loop over horizontal axis of the output volume
                    for c in range(C):           # loop over the channels of the output volume
                        # Find the corners of the current "slice"
                        vert_start = h * self.stride
                        vert_end = vert_start + self.filterSize
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.filterSize
                        # Use the corners to define the slice from x_pad
                        x_slice = x_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        d_in_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c]*d_Out[i,h,w,c]
                        grads['W'][:,:,:,c] += x_slice * d_Out[i,h,w,c]
                        grads['b'][:,:,:,c] += d_Out[i,h,w,c]
            # Set the ith training example's d_In to the unpaded d_in_pad
            if pad > 0:
                dIn[i, :, :, :] = d_in_pad[pad:-pad,pad:-pad,:]
            else:
                dIn[i, :, :, :] = d_in_pad
        return dIn, grads

class pooling2D(neuralNetLayer):
    def __init__(self, inputShape, filterSize, stride, mode='max'):
        assert mode in ['max', 'avg']
        height = int(1 + (inputShape[0] - filterSize) / stride)
        width = int(1 + (inputShape[1] - filterSize) / stride)
        channels = inputShape[-1]
        shape = ((height,width,channels), inputShape)
        neuralNetLayer.__init__(self, shape)
        
        self.trainable = False
        self.filterSize = filterSize
        self.stride = stride
        self.mode = mode
        self.cache = []
    
    def fwdPass(self, X, inference=False):
        self.cache = []
        (obs, H_prev, W_prev, C_prev) = X.shape
        H, W = self.shape[0][:2]
        C = C_prev
        
        A = np.zeros((obs, H, W, C))
        
        for i in range(obs):
            for h in range(H):
                for w in range(W):
                    for c in range (C):
                        # Find the corners of the current "slice"
                        vert_start = h * self.stride
                        vert_end = vert_start + self.filterSize
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.filterSize
                        # Use the corners to define the current slice on the ith training example of X, channel c
                        x_slice = X[i,vert_start:vert_end,horiz_start:horiz_end,c]
                        # Compute the pooling operation on the slice, based on the mode
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(x_slice)
                        elif self.mode == "avg":
                            A[i, h, w, c] = np.mean(x_slice)
        
        self.cache.append(X)
        return A
    
    def bwdPass(self, d_out):
        def createWindowMask(x):
            mask = x == np.max(x)
            return mask
        def distributeVal(d_o):
            H = W = self.filterSize
            average = 1 / (H * W)
            return d_o * average * np.ones((H, W))
        
        X = self.cache[0]
        obs, H_prev, W_prev, C_prev = X.shape
        obs, H, W, C = d_out.shape
        d_in = np.zeros((obs, H_prev, W_prev, C_prev))
        
        for i in range(obs):
            # select training example from X
            x = X[i,:,:,:]
            for h in range(H):                   # loop on the vertical axis
                for w in range(W):               # loop on the horizontal axis
                    for c in range(C):           # loop over the channels (depth)
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * self.stride
                        vert_end = vert_start + self.filterSize
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.filterSize
                        
                        # Compute the backward propagation in both modes.
                        if self.mode == "max":
                            # Use the corners and "c" to define the current slice from X
                            x_slice = x[vert_start:vert_end,horiz_start:horiz_end, c]
                            # Create the mask from x_slice
                            mask = createWindowMask(x_slice)
                            # Set d_in to be d_in + (the mask multiplied by the correct entry of d_out)
                            d_in[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*d_out[i,h,w,c]
                        elif self.mode == "avg":
                            # Get the value d_o from d_out
                            d_o = d_out[i,h,w,c]
                            # Distribute it to get the correct slice of d_in. i.e. Add the distributed value of o
                            d_in[i, vert_start: vert_end, horiz_start: horiz_end, c] += distributeVal(d_o)
        
        return d_in
    
    def paramUpdate(self):
        raise AttributeError( "'Pooling' layer has no method 'paramUpdate'." )

#intended for volumes (obs, 1, 1, featureMaps), i.e. dense layer
class flatten(neuralNetLayer):
    def __init__(self, shape):
        neuralNetLayer.__init__(self, shape)
        self.trainable = False
    
    def fwdPass(self, x, inference=False):
        obs, _, _, featureMaps = x.shape
        return np.reshape(x, (obs, featureMaps)).T
    
    def bwdPass(self, dOut):
        featureMaps, obs = dOut.shape
        return np.reshape(dOut.T, (obs, 1, 1, featureMaps))
    
    def paramUpdate(self):
        raise AttributeError( "'Flatten' layer has no method 'paramUpdate'." )
