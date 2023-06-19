import numpy as np
import argparse
import pathlib
import sys

class denseLayer:
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.01 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# Activation

class activationSigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)

class activationTanh:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output * self.output)

# Loss
# y needs to be in the form of masks rather than one hot encoded

class loss:
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)

        return dataLoss

class crossEntropy(loss):
    def forward(self, yPred, y):
        samples = len(yPred)
        yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)

        correctConfidences = yPredClipped[range(samples), y]

        return -np.log(correctConfidences)

    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])

        yTrue = np.eye(labels)[y]

        self.dinputs = -yTrue / dvalues
        self.dinputs = self.dinputs / samples

class meanSquare(loss):
    def forward(self, yPred, y):
        labels = len(yPred[0])
        yTrue = np.eye(labels)[y]
        return np.mean((yTrue - yPred) ** 2, axis = -1)
    
    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])

        yTrue = np.eye(labels)[y]

        self.dinputs = -2 * (yTrue - dvalues) / labels
        self.dinputs = self.dinputs / samples

# Accuracy / Error
def accuracy(yPred, y):
    pred = np.argmax(yPred)
    accuracy = np.mean(pred == y)

    return 100 * (1 - accuracy)

# Optimization Algorithms

class gd:
    def __init__(self, learningRate = 1.0):
        self.learningRate = learningRate
    
    def updateParams(self, layer):
        layer.weights += -self.learningRate * layer.dweights
        layer.biases += -self.learningRate * layer.dbiases

class momentum:
    pass

class nag:
    pass

class adam:
    def __init__(self, learningRate = 0.01, decay = 0., epsilon = 1e-7, beta1 = 0.9, beta2 = 0.999):
        self.learningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    def preUpdateParams(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):
        if not hasattr(layer, 'weightCache'):
            layer.weightMomentums = np.zeros_like(layer.weights)
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasMomentums = np.zeros_like(layer.biases)
            layer.biasCache = np.zeros_like(layer.biases)

        layer.weightMomentums = self.beta1 * layer.weightMomentums + (1 - self.beta1) * layer.dweights
        layer.biasMomentums = self.beta1 * layer.biasMomentums + (1 - self.beta1) * layer.dbiases

        weightMomentumsCorrected = layer.weightMomentums / (1 - self.beta1 ** (self.iterations + 1))
        biasMomentumsCorrected = layer.biasMomentums / (1 - self.beta1 ** (self.iterations + 1))

        layer.weightCache = self.beta2 * layer.weightCache + (1 - self.beta2) * layer.dweights ** 2
        layer.biasCache = self.beta2 * layer.biasCache + (1 - self.beta2) * layer.dweights ** 2

        weightCacheCorrected = layer.weightCache / (1 - self.beta2 ** (self.iterations + 1))
        biasCacheCorrected = layer.biasCache / (1 - self.beta2 ** (self.iterations + 1))

        layer.weights += -self.currentLearningRate * weightMomentumsCorrected / (np.sqrt(weightCacheCorrected) + self.epsilon)
        layer.biases += -self.currentLearningRate * biasMomentumsCorrected / (np.sqrt(biasCacheCorrected) + self.epsilon)

    def postUpdateParams(self):
        self.iterations += 1

def main(learningRate, momentumVal, numHidden, sizes, activationFunction, lossFunction, optimizer, decay, train, test, saveDir, exptDir):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', help="initial learning rate for gd based algorithms", type=float)
    parser.add_argument('--momentum', help="momentum to be used by momentum based algorithms", type=float)
    parser.add_argument('--num_hidden', help="number of hidden layers", type=int)
    parser.add_argument('--sizes', help="comma separated list for the size of each hidden layer", type=str)
    parser.add_argument('--activation', help="[tanh / sigmoid]", default="sigmoid", type=str)
    parser.add_argument('--loss', help="[sq / ce]", default="ce", type=str)
    parser.add_argument('--opt', help="[gd / momentum / nag / adam]", default="gd", type=str)
    parser.add_argument('--batch_size', help="[1 / multiples of 5]", default=1, type=int)
    parser.add_argument('--anneal', help="[true / false]", default= "true", type=str)
    parser.add_argument('--save_dir', help="path to pickled model", type=pathlib.Path)
    parser.add_argument('--expt_dir', help="path to log files", type=pathlib.Path)
    parser.add_argument('--train', help="training data", type=str)
    parser.add_argument('--test', help="test data", type=str)

    args = parser.parse_args()

    learningRate = args.lr
    momentumVal = args.momentum
    numHidden = args.num_hidden
    sizes = list(map(int, args.sizes.split(',')))
    activation = args.activation
    lossFunc = args.loss
    opt = args.opt
    batchSize = args.batch_size
    anneal = args.anneal
    saveDir = args.save_dir
    exptDir = args.expt_dir
    train = args.train
    test = args.test

    if len(sizes) != numHidden:
        print("Length of sizes should be the same as num_hidden")
        sys.exit(0)

    if activation != "sigmoid" and activation != "tanh":
        print("activation needs to be one of sigmoid / tanh")
        sys.exit(0)

    if lossFunc != "sq" and lossFunc != "ce":
        print("loss should be one of sq / ce")
        sys.exit(0)

    if opt not in ["gd", "momentum", "nag", "adam"]:
        print("opt not valid")
        sys.exit(0)

    if batchSize != 1 and batchSize % 5 != 0:
        print("batch_size should be either 1 or multiple of 5")
        sys.exit(0)

    if anneal not in ["true", "false"]:
        print("anneal should be true / false")
        sys.exit(0)
        
    if activation == "sigmoid":
        activationFunction = activationSigmoid()
    else:
        activationFunction = activationTanh()

    if lossFunc == "sq":
        lossFunction = meanSquare()
    else:
        lossFunction = crossEntropy()

    if opt == "gd":
        optimizer = gd(learningRate)
    elif opt == "momentum":
        optimizer = momentum()
    elif opt == "nag":
        optimizer = nag()
    else:
        optimizer = adam(learningRate)

    if anneal == "true":
        decay = .5
    else:
        decay = 1

    main(learningRate, momentumVal, numHidden, sizes, activationFunction, lossFunction, optimizer, decay, train, test, saveDir, exptDir)
    