import numpy as np

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
        return np.square(yPred - yTrue).mean(axis=1)
    
    def backward(self, dvalues, y):
        self.dinputs = -2 * (y - dvalues)

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

    
# class momentum:

# class nag:

# class adam:
