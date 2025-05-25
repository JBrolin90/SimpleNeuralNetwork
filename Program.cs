using BackPropagation.NNLib;
// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, BackPropagation learners!");

double[] inputs = [0, 0.5, 1];

double[] observed = [0, 1, 0];


double[][][] weights = new double[2][][];
weights[0] = new double[2][];
weights[0][0] = [3.34];
weights[0][1] = [-3.53];
weights[1] = new double[1][];
weights[1][0] = [-1.22, -2.3];

double[][] biases = new double[2][];
biases[0] = [-1.43, 0.57];
biases[1] = [3.0];

Func<double, double>[] activationFunctions = [
    Node.SoftPlus, // Activation function for the first layer
    Node.UnitActivation // Activation function for the second layer
];

NeuralNetwork nn = new(new LayerFactory(), new NodeFactory(),
    weights, biases, activationFunctions, 0.01);
double[] output0 = nn.Predict([0]);
double[] output1 = nn.Predict([0.5]);
double[] output2 = nn.Predict([1.0]);
Console.WriteLine($"Outputs: {output0[0]}, {output1[0]}, {output2[0]}");
// Test the neural network with the provided inputs
TestNeuralNetwork testNN = new(new LayerFactory(), new NodeFactory(),
    weights, biases, 0.01, activationFunctions);
testNN.Test(inputs, observed);
Console.WriteLine($"SSR: {testNN.SSR}");