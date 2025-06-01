using System;
using System.ComponentModel;
using System.Net.WebSockets;

namespace BackPropagation.NNLib;

public struct NodeSteps
{
    public double[] WeightSteps;
    public double BiasStep;
}

public class TestNeuralNetwork : NeuralNetwork
{
    public double SSR = 0;
    public double dSSR = 0;
    public NodeSteps[][] nodeSteps = Array.Empty<NodeSteps[]>();

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
            SSR += 0.5 * Math.Pow(expectedOutputs[i] - predictions[i][0], 2);
            dSSR = predictions[i][0] - expectedOutputs[i];
            BackPropagate(dSSR);
        }
        UpdateWeightsAndBiases();

    }

    public void BackPropagate(double dSSR)
    {
        NodeSteps[][] nodeSteps = new NodeSteps[Layers.Length][];

        int i = 0;
        foreach (var layer in Layers)
        {
            nodeSteps[i++] = layer.Backward(nodeSteps[i], dSSR);
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
                    // double delta = stepSizes[j1][k1][l];
                    //Weigths[j1][k1][l] -= delta * 0.1; //LearningRate;
                }
            }
        }
    }
}
