
using System;
using BackPropagation.NNLib;

var nodeFactory = new NeuronFactory();
var inputProcessorFactory = new InputProcessorFactory();
double[][] weights = { 
    new double[] { 0.5, 0.2 }, 
    new double[] { -0.3, 0.8 } 
};
double[][] biases = { 
    new double[] { 0.1 }, 
    new double[] { -0.2 } 
};
var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Sigmoid);

double[] inputs1 = { 1.0, 2.0 };
double[] outputs1 = layer.Forward(inputs1);
Console.WriteLine($"Input1: [{string.Join(", ", inputs1)}]");
Console.WriteLine($"Output1: [{string.Join(", ", outputs1)}]");

double[] inputs2 = { -1.0, 0.5 };
double[] outputs2 = layer.Forward(inputs2);
Console.WriteLine($"Input2: [{string.Join(", ", inputs2)}]");
Console.WriteLine($"Output2: [{string.Join(", ", outputs2)}]");

// Check calculations for node 1
double node1_sum1 = 1.0 * 0.5 + 2.0 * 0.2 + 0.1;
double node1_sum2 = -1.0 * 0.5 + 0.5 * 0.2 + 0.1;
Console.WriteLine($"Node1 sum1: {node1_sum1}, sigmoid: {1.0 / (1.0 + Math.Exp(-node1_sum1))}");
Console.WriteLine($"Node1 sum2: {node1_sum2}, sigmoid: {1.0 / (1.0 + Math.Exp(-node1_sum2))}");

