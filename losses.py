import numpy as np
from activations import *

#Note: labels ARE NOT one-hot encoded!!
def softmaxWithXentropy(logits, labels, recurrent=False):
    obs = logits.shape[1]
    if recurrent:
        steps = logits.shape[2]
        cost = 0
        dError = np.zeros_like(logits)
        for t in range(steps):
            probs = softmax(logits[:,:,t])
            logLikelihood = -np.log(probs[labels[:,t],range(obs)])
            cost += np.sum(logLikelihood) / obs
            dError[:,:,t] = probs.copy()
            dError[labels[:,t],range(obs),t] -= 1
            dError[:,:,t] /= obs
    else:
        probs = softmax(logits)
        logLikelihood = -np.log(probs[labels,range(obs)])
        cost = np.sum(logLikelihood) / obs
        dError = probs.copy()
        dError[labels, range(obs)] -= 1
        dError /= obs
    return cost, dError

def xentropy(probs, labels, recurrent=False):
    obs = probs.shape[1]
    if recurrent:
        steps = probs.shape[2]
        cost = 0
        dError = np.zeros_like(probs)
        for t in range(steps):
            logLikelihood = -np.log(probs[labels[:,t],range(obs),t])
            cost += np.sum(logLikelihood) / obs
            dError[labels[:,t],range(obs),t] -= 1/probs[labels[:,t], range(obs),t]
            dError[:,:,t] /= obs
    else:
        logLikelihood = -np.log(probs[labels,range(obs)])
        cost = np.sum(logLikelihood) / obs
        dError = np.zeros_like(probs)
        dError[labels, range(obs)] -= 1/probs[labels, range(obs)]
        dError /= obs
    return cost, dError

def mse(predicted, labels, recurrent=False):
    obs = predicted.shape[0] * predicted.shape[1]
    if recurrent:
        steps = predicted.shape[2]
        cost = 0
        dError = np.zeros_like(predicted)
        for t in range(steps):
            sqErrors = np.power(predicted[:,:,t] - labels[:,:,t], 2)
            cost += 0.5 * np.sum(sqErrors) / obs
            dError[:,:,t] =  predicted[:,:,t] - labels[:,:,t]
            dError[:,:,t] /= obs
    else:
        sqErrors = np.power(predicted - labels, 2)
        cost = 0.5 * np.sum(sqErrors) / obs
        dError = predicted - labels
        dError /= obs
    return cost, dError

#Note: labels ARE NOT one-hot encoded!!
def hingeLoss(scores, targets, recurrent=False):
	obs = scores.shape[1]
	marginThreshold = 1
	if recurrent:
		steps = scores.shape[2]
		cost = 0
		dError = np.zeros_like(scores)
		for t in range(steps):
			trueScores = scores[targets[:,t], range(obs),t]
			margins = np.maximum(0, scores[:,:,t] - trueScores + marginThreshold)
			margins[targets[:,t], range(obs)] = 0
			cost += np.sum(margins) / obs
			temp = margins
			temp[temp > 0] = 1
			dError[:,:,t] = temp
			colSum = np.sum(dError[:,:,t], axis=0)
			dError[targets[:,t], range(obs),t] -= colSum
			dError[:,:,t] /= obs

	else:
		trueScores = scores[targets, range(obs)]
		margins = np.maximum(0, scores - trueScores + marginThreshold)
		margins[targets, range(obs)] = 0
		cost = np.sum(margins) / obs
		dError = margins
		dError[dError > 0] = 1
		colSum = np.sum(dError, axis=0)
		dError[targets, range(obs)] -= colSum
		dError = dError / obs
	return cost, dError
