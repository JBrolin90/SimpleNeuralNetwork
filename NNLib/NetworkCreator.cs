using System;
using System.Globalization;
using System.Security;

namespace BackPropagation.NNLib;

public class NetworkCreator
{
    public double[][][] Weights { get; set; }
    public double[][][] Biases { get; set; }
    public Func<double,double>[] ActivationFunctions { get; set; }

    public double[][] Ys { get; set; }
    public NetworkCreator(int[] layerSizes, Func<double,double>[] activationFunctions)
    {
        Weights = new double[layerSizes.Length][][];
        Weights[0] = [];  // Input layer has no weights
        Weights[^1] = []; // Output layer has no weights
        ActivationFunctions = activationFunctions;
        for (int i = 1; i < layerSizes.Length - 1; i++) // Exclude output layer
        {
            Weights[i] = new double[layerSizes[i]][];
            for (int j = 0; j < layerSizes[i]; j++) // Nodes
            {
                Weights[i][j] = new double[layerSizes[i-1]]; // Size should be previous layer size
                for (int k = 0; k < layerSizes[i-1]; k++) // Inputs from previous layer
                {
                    Weights[i][j][k] = 0;
                }
            }
        }
        Biases = new double[layerSizes.Length][][];
        Biases[0] = []; // Input layer has no biases
        Biases[^1] = []; // Output layer has no biases
        for (int i = 1; i < layerSizes.Length - 1; i++) // Exclude output layer
        {
            Biases[i] = new double[layerSizes[i]][];
            for (int j = 0; j < layerSizes[i]; j++) // Nodes
            {
                Biases[i][j] = [0]; // Each node has one bias
            }
        }
        // Initialize Ys based on layer sizes
        Ys = new double[layerSizes.Length][];
        for (int i = 0; i < layerSizes.Length; i++)
        {
            Ys[i] = new double[layerSizes[i]];
        }
    }

    public void RandomizeWeights()
    {
        for (int i = 1; i < Weights.Length-1; i++) // Exclude output layer
        {
            for (int j = 0; j < Weights[i].Length; j++)
            {
                for (int k = 0; k < Weights[i][j].Length; k++)
                {
                    Weights[i][j][k] = new Random().NextDouble()*10-5;
                }
            }
        }
    }

    public NeuralNetworkTrainer CreateNetwork()
    {
        NeuralNetworkTrainer n = new(new LayerFactory(), new NodeFactory(),
                        Weights, Biases, Ys,
                        0.01, ActivationFunctions);
        return n;
    }
}
