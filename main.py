from NeuralNetwork import NeuralNetwork

neuralNet = NeuralNetwork(
    layers=(2, 2, 2, 1),
    learning_rate=1E-3,
    momentum=.04,
)

print(neuralNet.w1)
