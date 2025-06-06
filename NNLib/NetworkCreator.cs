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
        Weights[0] = [];  
        Weights[^1] = []; 
        ActivationFunctions = activationFunctions;
        for (int i = 1; i < layerSizes.Length - 1; i++) //Layers
        {
            Weights[i] = new double[layerSizes[i]][];
            for (int j = 0; j < layerSizes[i]; j++) // Nodes
            {
                Weights[i][j] = new double[i];
                if (i > 0)
                {
                    for (int k = 0; k < layerSizes[i-1]; k++) // Inputs from previous layer
                    {
                        Weights[i][j][k] = 0;
                    }
                }
            }
        }
        Biases = new double[layerSizes.Length][][];
        Biases[0] = []; 
        Biases[^1] = []; 
        for (int i = 1; i < layerSizes.Length - 1; i++) //Layers
        {
            Biases[i] = new double[layerSizes[i]][];
            for (int j = 0; j < layerSizes[i]; j++) // Nodes
            {
                for (int k = 0; k < layerSizes[i]; k++) // Inputs
                {
                    Biases[i][j] = [0]; //Allways one bias
                }
            }
            Biases[i][0] = [0];
        }
        Ys = [[0, 0], [0, 0], [0, 0], [0, 0]];
    }

    public void RandomizeWeights()
    {
        for (int i = 1; i < Weights.Length-1; i++)
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
