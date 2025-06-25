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
    public NetworkCreator(int inputs, int[] layerSizes, Func<double,double>[] activationFunctions)
    {
        Weights = new double[layerSizes.Length][][];
        ActivationFunctions = activationFunctions;
        for (int i = 0; i < layerSizes.Length; i++) 
        {
            Weights[i] = new double[layerSizes[i]][];
            for (int j = 0; j < layerSizes[i]; j++) // Nodes
            {
                inputs = i > 0 ? layerSizes[i - 1] : inputs;
                Weights[i][j] = new double[inputs]; // Size should be previous layer size
                for (int k = 0; k < inputs; k++) // Inputs from previous layer
                {
                    Weights[i][j][k] = 0;
                }
            }
        }
        Biases = new double[layerSizes.Length][][];
        for (int i = 0; i < layerSizes.Length; i++) 
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

    public void RandomizeWeights(double from, double to)
    {
        for (int i = 0; i < Weights.Length; i++) // Exclude output layer
        {
            for (int j = 0; j < Weights[i].Length; j++)
            {
                for (int k = 0; k < Weights[i][j].Length; k++)
                {
                    Weights[i][j][k] = new Random().NextDouble()*(to-from)-(to-from)/2;
                }
            }
        }
    }

    public NeuralNetworkTrainer CreateNetwork(double trainingRate = 0.001)
    {
        NeuralNetworkTrainer n = new(new LayerFactory(), new NodeFactory(),
                        Weights, Biases, Ys,
                        trainingRate, ActivationFunctions);
        return n;
    }
}
