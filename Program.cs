using BackPropagation.NNLib;
// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, BackPropagation learners!");

double[] inputs = [0, 0.5, 1];
double[] observed = [0, 1, 0];

double[][][] weights = [[],[[3.34], [-3.53]], [[0.36, 0.63]],[]];
double[][][] biases = [[], [ [-1.43], [0.57]], [[0]], []];
double[][] ys = [[0,0],[0,0],[0, 0], [0, 0]];


Func<double, double>[] activationFunctions = [
    ActivationFunctions.Unit, // Activation function for the second layer
    ActivationFunctions.SoftPlus, // Activation function for the first layer
    ActivationFunctions.Unit, // Activation function for the second layer
    ActivationFunctions.Unit // Activation function for the second layer
];


// NeuralNetwork nn = new(new LayerFactory(), new NodeFactory(),
//     weights, biases, ys, activationFunctions, 0.01);


// double[] output0 = nn.Predict([0]);
// double[] output1 = nn.Predict([0.5]);
// double[] output2 = nn.Predict([1.0]);
// Console.WriteLine($"Outputs: {output0[0]}, {output1[0]}, {output2[0]}");



TestNeuralNetwork testNN = new(new LayerFactory(), new NodeFactory(),
    weights, biases, ys, 0.01, activationFunctions);
for (int i = 0; i < 1; i++)
{
    testNN.Test(inputs, observed);
}
Console.WriteLine($"SSR: {testNN.SSR}");