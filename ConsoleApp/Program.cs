using BackPropagation;
using BackPropagation.NNLib;
// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, BackPropagation learners!");

// Test original SimpleAdder implementation
const int epochs = 5000;
double[][] samples = [[0], [0.5], [1]];
double[][] observed = [[0], [1], [0]];

double[][][] weights = [
    [[2.74], [-1.13]],
    [[0.36, 0.63]],
    ];
double[][][] biases = [[[0], [0]], [[0]]];
double[][] ys = [[0, 0], [0, 0], [0, 0], [0, 0]];


Func<double, double>[] activationFunctions = [
    ActivationFunctions.SoftPlus, // Activation function for the second layer
    ActivationFunctions.Unit, // Activation function for the first layer
    ActivationFunctions.Unit, // Activation function for the second layer
    ActivationFunctions.Unit // Activation function for the second layer
];


// NeuralNetwork nn = new(new LayerFactory(), new NodeFactory(),
//     weights, biases, ys, activationFunctions, 0.01);


// double[] output0 = nn.Predict([0]);
// double[] output1 = nn.Predict([0.5]);
// double[] output2 = nn.Predict([1.0]);
// Console.WriteLine($"Outputs: {output0[0]}, {output1[0]}, {output2[0]}");



NeuralNetworkTrainer nn1 = new(new LayerFactory(), new NodeFactory(),
    weights, biases, ys, 0.01, activationFunctions);

double[] output0 = nn1.Predict([0]);
double[] output1 = nn1.Predict([0.5]);
double[] output2 = nn1.Predict([1.0]);
Console.WriteLine($"Outputs: {output0[0]}, {output1[0]}, {output2[0]} Sum = {output0[0]+output1[0]+output2[0]}");



// for (int i = 0; i < epochs; i++)
// {
//     nn1.Train(samples, observed);
// }
// Console.WriteLine($"SSR nn1: {nn1.SSR[0]}");

// NetworkCreator creator = new(1, [2, 1], activationFunctions);
// creator.RandomizeWeights();
// NeuralNetworkTrainer nn2 = creator.CreateNetwork();

// for (int i = 0; i < epochs; i++)
// {
//     nn2.Train(samples, observed);
// }
// Console.WriteLine($"SSR nn2: {nn2.SSR[0]}");

new Algebra().DoIt();

Console.WriteLine("\n" + new string('=', 50));
new AlgebraTest().DoIt();