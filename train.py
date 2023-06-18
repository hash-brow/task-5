import numpy as np

class denseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Activation

class activationSigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(inputs))

class activationTanh:
    def forward(self, inputs):
        self.output = np.tanh(inputs)

# Loss
# y needs to be in the form of masks rather than one hot encoded

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

class crossEntropy(Loss):
    def forward(self, y_pred, y):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = y_pred_clipped[range(samples), y]

        return -np.log(correct_confidences)

class meanSquare(Loss):
    def forward(self, y_pred, y):
        samples = y_pred.shape[0]
        y_true = np.zeros((samples, y_pred.shape[1]))
        for idx in range(samples):
            y_true[idx][y[idx]] = 1

        return np.square(y_pred - y_true).mean(axis=1)
    
# Accuracy / Error
def accuracy(y_pred, y):
    pred = np.argmax(y_pred)
    accuracy = np.mean(pred == y)

    return 100 * (1 - accuracy)

# Optimization Algorithms

# class gd:

# class momentum:

# class nag:

# class adam:
