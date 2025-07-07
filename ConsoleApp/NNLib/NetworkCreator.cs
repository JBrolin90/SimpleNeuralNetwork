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
        double NextDouble(double _) => from + new Random().NextDouble() * (to - from);
        ApplyOn3dArr(Weights, NextDouble);

    }

    public static void ActOn3dArr(double[][][] arr, Action<double> action)
    {
        if (arr == null)
            throw new ArgumentNullException(nameof(arr));
        if (action == null)
            throw new ArgumentNullException(nameof(action));

        int layerCount = arr.Length;
        for (int i = 0; i < layerCount; i++)
        {
            int nodeCount = arr[i].Length;
            for (int j = 0; j < nodeCount; j++)
            {
                int inputCount = arr[i][j].Length;
                for (int k = 0; k < inputCount; k++)
                {
                    action(arr[i][j][k]);
                }
            }
        }
    }
    public static void ApplyOn3dArr(double[][][] arr, Func<double, double> func)
    {
        if (arr == null)
            throw new ArgumentNullException(nameof(arr));
        if (func == null)
            throw new ArgumentNullException(nameof(func));

        int layerCount = arr.Length;
        for (int i = 0; i < layerCount; i++)
        {
            int nodeCount = arr[i].Length;
            for (int j = 0; j < nodeCount; j++)
            {
                int inputCount = arr[i][j].Length;
                for (int k = 0; k < inputCount; k++)
                {
                    arr[i][j][k] = func(arr[i][j][k]);
                }
            }
        }
    }

    public INeuralNetwork CreateNetwork()
    {
        // Create deep copies of weights, biases, and ys to ensure independence
        var weightsCopy = DeepCopy3DArray(Weights);
        var biasesCopy = DeepCopy3DArray(Biases);
        var ysCopy = DeepCopy2DArray(Ys);
        
        NeuralNetwork n = new(new LayerFactory(), new NeuronFactory(), new InputProcessorFactory(),
                        weightsCopy, biasesCopy, ysCopy,
                        ActivationFunctions);
        return n;
    }
    
    private static double[][][] DeepCopy3DArray(double[][][] source)
    {
        if (source == null) return new double[0][][];
        
        var copy = new double[source.Length][][];
        for (int i = 0; i < source.Length; i++)
        {
            if (source[i] != null)
            {
                copy[i] = new double[source[i].Length][];
                for (int j = 0; j < source[i].Length; j++)
                {
                    if (source[i][j] != null)
                    {
                        copy[i][j] = new double[source[i][j].Length];
                        Array.Copy(source[i][j], copy[i][j], source[i][j].Length);
                    }
                }
            }
        }
        return copy;
    }
    
    private static double[][] DeepCopy2DArray(double[][] source)
    {
        if (source == null) return new double[0][];
        
        var copy = new double[source.Length][];
        for (int i = 0; i < source.Length; i++)
        {
            if (source[i] != null)
            {
                copy[i] = new double[source[i].Length];
                Array.Copy(source[i], copy[i], source[i].Length);
            }
        }
        return copy;
    }
}
