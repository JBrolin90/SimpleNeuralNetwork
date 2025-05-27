using System;
using System.ComponentModel;
using System.Net.WebSockets;

namespace BackPropagation.NNLib;

public class TestNeuralNetwork : NeuralNetwork
{
    public double SSR = 0;
    public double dSSR = 0;
    public double db3 = 0;
    public double dw3 = 0;
    public double dw4 = 0;

    public double[][][] stepSizes = [
        [
            [0.0],
            [0.0]
        ],
        [
            [0.0, 0.0]
        ]
    ];

    public TestNeuralNetwork(ILayerFactory LayerFactory, INodeFactory NodeFactory,
                        double[][][] weights, double[][][] biases, double[][] ys,
                        double learningRate = 0.01, Func<double, double>[]? activationFunctions = null)
                        : base(LayerFactory, NodeFactory, weights, biases, ys, activationFunctions, learningRate)
    { }

    public void Test(double[] inputs, double[] expectedOutputs)
    {
        double[][] predictions = new double[inputs.Length][];
        SSR = 0; dSSR = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            predictions[i] = Predict([inputs[i]]);
            SSR += Math.Pow(expectedOutputs[i] - predictions[i][0], 2);
            dSSR = -2 * (expectedOutputs[i] - predictions[i][0]);
            Descent(dSSR);
            db3 += dSSR * 1;
        }
        UpdateWeightsAndBiases();
        // Weigths[1][0][0] -= dw3 * 0.1; // Update the weight for the output Layer
        // Weigths[1][0][1] -= dw3 * 0.1; // Update the weight for the output Layer
        Biases[1][0][0] -= db3 * 0.1; // Update the bias for the output layer


        Console.WriteLine($"Inputs: {string.Join(", ", inputs)}");
        Console.WriteLine($"Expected Outputs: {string.Join(", ", expectedOutputs)}");
        Console.WriteLine($"Predicted Outputs: {string.Join(", ", predictions.Select(arr => string.Join(";", arr)))}");
        Console.WriteLine();
    }

    public void Descent(double dSSR)
    {
        double[][][] gradients = new double[Layers.Length][][];
        int i = 0;
        foreach (var layer in Layers)
        {
            gradients[i++] = layer.Descent(dSSR);
        }
        int j = 0;
        foreach (var layer in gradients)
        {
            int k = 0;
            foreach (var node in layer)
            {
                for (int l = 0; l < node.Length; l++)
                {
                    double r = node[l];
                    stepSizes[j][k][l] += r;
                }
                k++;
            }
            j++;
        }
    }

    private void UpdateWeightsAndBiases()
    {
        // Update weights and biases
        for (int j1 = 0; j1 < Weigths.Length; j1++)
        {
            for (int k1 = 0; k1 < Weigths[j1].Length; k1++)
            {
                for (int l = 0; l < Weigths[j1][k1].Length; l++)
                {
                    double delta = stepSizes[j1][k1][l];
                    Weigths[j1][k1][l] -= delta * 0.1; //LearningRate;
                }
            }
        }
    }
}
