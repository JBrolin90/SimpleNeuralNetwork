using System;
using System.ComponentModel;
using System.Net.WebSockets;

namespace BackPropagation.NNLib;

public class NodeSteps(int weightCount)
{
    public double[] WeightSteps = new double[weightCount];
    public double BiasStep = 0;
}

public class NeuralNetworkTrainer : NeuralNetwork
{
    public double SSR = 0;
    public double dSSR = 0;

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

    public void Train(double[] inputs, double[] expectedOutputs)
    {
        // inputs is a single sample with multiple features
        // expectedOutputs is the expected output for this single sample
        SSR = 0; dSSR = 0;
        PrepareBackPropagation();
        
        var prediction = Predict(inputs);
        SSR += Math.Pow(expectedOutputs[0] - prediction[0], 2);
        dSSR = -2 * (expectedOutputs[0] - prediction[0]);
        BackPropagate(dSSR);
        
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

    public void BackPropagate(double dSSR)
    {
        int i = 0;
        foreach (var layer in Layers)
        {
            layer.Backward(dSSR, NodeSteps[i]);
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
