using System;
using System.ComponentModel;
using System.Globalization;
using System.Net.WebSockets;

namespace BackPropagation.NNLib;

public class NodeSteps(int weightCount)
{
    public double[] WeightSteps = new double[weightCount];
    public double BiasStep = 0;
}

public class NeuralNetworkTrainer : NeuralNetwork
{
    public double[] SSR = [];
    public double[] dSSR = [];

    public double LearningRate = 0;
    public NodeSteps[][] NodeSteps = Array.Empty<NodeSteps[]>();

    public NeuralNetworkTrainer(ILayerFactory LayerFactory, INodeFactory NodeFactory,
                        double[][][] weights, double[][][] biases, double[][] ys,
                        double learningRate = 0.01, Func<double, double>[]? activationFunctions = null)
                        : base(LayerFactory, NodeFactory, weights, biases, ys, 
                              activationFunctions ?? CreateDefaultActivationFunctions(weights.Length))
    {
        LearningRate = learningRate;
    }

    private static Func<double, double>[] CreateDefaultActivationFunctions(int layerCount)
    {
        var functions = new Func<double, double>[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            functions[i] = BackPropagation.NNLib.ActivationFunctions.Unit;
        }
        return functions;
    }

    public void Train(double[][] trainingData, double[][] expectedOutputs)
    {
        double[][] predictions = new double[trainingData.Length][];
        int outputCount = Layers[^1].Nodes.Length;
        SSR = new double[outputCount];
        dSSR = new double[outputCount];
        PrepareBackPropagation();
        for (int i = 0; i < trainingData.Length; i++)
        {
            predictions[i] = Predict(trainingData[i]);
            for (int j = 0; j < outputCount; j++)
            {
                SSR[j] = Math.Pow(expectedOutputs[i][j] - predictions[i][j], 2);
                dSSR[j] = -2 * (expectedOutputs[i][j] - predictions[i][j]);
            }

            BackPropagate(dSSR);
        }


        UpdateWeightsAndBiases();
    }

    

    public void PrepareBackPropagation()
    {
        NodeSteps = new NodeSteps[Layers.Length][];
        int i = 0;
        foreach (var layer in Layers)
        {
            NodeSteps[i] = new NodeSteps[layer.Nodes.Length];
            for (int j = 0; j < layer.Nodes.Length; j++)
            {
                NodeSteps[i][j] = new NodeSteps(layer.Nodes[j].Weights.Length);
            }
            i++;
        }
    }

    public void BackPropagate(double[] dSSR)
    {
        int i = 0;
        foreach (var layer in Layers)
        {
            layer.Backward(dSSR[0], NodeSteps[i]);
            i++;
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
                    double deltaW = NodeSteps[j1][k1].WeightSteps[l];
                    Weigths[j1][k1][l] -= deltaW * LearningRate;
                }
                double deltaB = NodeSteps[j1][k1].BiasStep;
                Biases[j1][k1][0] -= deltaB * LearningRate;
            }
        }
    }
}
