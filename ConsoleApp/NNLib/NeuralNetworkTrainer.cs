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

    public Func<double[][], double[][], double[]> LossFunction = LossFunctions.SumSquaredError;
    public Func<double[][], double[][], double[]> LossFunctionD = LossFunctions.SumSquaredErrorDerivative;
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

    double[] loss = [];
    double[] dLoss = [];
    public double[] TrainOneEpoch(double[][] trainingData, double[][] expectedOutputs)
    {
        int outputCount = Layers[^1].Nodes.Length;

        double[][] predictions = new double[trainingData.Length][];
        for (int i = 0; i < trainingData.Length; i++)
        {
            // Forward pass
            predictions[i] = Predict(trainingData[i]);
        }
        loss = LossFunction(predictions, expectedOutputs);
        dLoss = LossFunctionD(predictions, expectedOutputs);
        PrepareBackPropagation();
        ResetGradients();
        PropagateBackwards(dLoss);  //BackPropagateRecursive(dSSR);
        UpdateWeightsAndBiases();

        return loss;
    }
    public double[] TrainOneEpoch(Sample[] trainingData)
    {
        int outputCount = Layers[^1].Nodes.Length;
        double[][] expectedOutputs = new double[trainingData.Length][];

        double[][] predictions = new double[trainingData.Length][];
        for (int i = 0; i < trainingData.Length; i++)
        {
            // Forward pass
            predictions[i] = Predict(trainingData[i].Xample);
            expectedOutputs[i] = [trainingData[i].Observed];
        }
        loss = LossFunction(predictions, expectedOutputs);
        dLoss = LossFunctionD(predictions, expectedOutputs);
        PrepareBackPropagation();
        ResetGradients();
        PropagateBackwards(dLoss);  //BackPropagateRecursive(dSSR);
        UpdateWeightsAndBiases();

        return loss;
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

    private void ResetGradients()
    {
        for (int i = 0; i < NodeSteps.Length; i++)
        {
            for (int j = 0; j < NodeSteps[i].Length; j++)
            {
                NodeSteps[i][j].BiasStep = 0;
                for (int k = 0; k < NodeSteps[i][j].WeightSteps.Length; k++)
                {
                    NodeSteps[i][j].WeightSteps[k] = 0;
                }
            }
        }
    }


    public void BackPropagateRecursive(double[] dSSR)
    {
        int i = 0;
        foreach (var layer in Layers)
        {
            layer.Backward(dSSR[0], NodeSteps[i]);
            i++;
        }
    }



    // Store node errors for each layer during backpropagation
    private double[][] nodeErrors = Array.Empty<double[]>();

    public void PropagateBackwards(double[] dLoss)
    {
        // Initialize error storage for all layers
        nodeErrors = new double[Layers.Length][];
        for (int i = 0; i < Layers.Length; i++)
        {
            nodeErrors[i] = new double[Layers[i].Nodes.Length];
        }

        // Start from output layer and work backwards
        for (int layerIndex = Layers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            var layer = Layers[layerIndex];

            if (layerIndex == Layers.Length - 1)
            {
                // Output layer: use direct error gradients and store them
                for (int nodeIndex = 0; nodeIndex < layer.Nodes.Length; nodeIndex++)
                {
                    var node = layer.Nodes[nodeIndex];
                    double error = dLoss[nodeIndex];
                    nodeErrors[layerIndex][nodeIndex] = error;

                    // Calculate gradients for this output node
                    CalculateNodeGradients(node, error, layerIndex, nodeIndex);
                }
            }
            else
            {
                // Hidden layers: calculate error from ALREADY CALCULATED next layer errors
                for (int nodeIndex = 0; nodeIndex < layer.Nodes.Length; nodeIndex++)
                {
                    var node = layer.Nodes[nodeIndex];
                    double error = CalculateErrorFromNextLayer(layerIndex, nodeIndex);
                    nodeErrors[layerIndex][nodeIndex] = error;

                    // Calculate gradients for this hidden node
                    CalculateNodeGradients(node, error, layerIndex, nodeIndex);
                }
            }
        }
    }

    private void CalculateNodeGradients(INode node, double error, int layerIndex, int nodeIndex)
    {
        // Get the layer inputs for gradient calculation
        var layer = Layers[layerIndex];
        double[] layerInputs = layer.Inputs ?? new double[0];

        // Calculate weight gradients
        for (int weightIndex = 0; weightIndex < node.Weights.Length; weightIndex++)
        {
            double input = (weightIndex < layerInputs.Length) ? layerInputs[weightIndex] : 0;
            double gradient = error * node.ActivationDerivative(node.Sum) * input;
            NodeSteps[layerIndex][nodeIndex].WeightSteps[weightIndex] += gradient;
            Console.WriteLine($"Gradient l{layerIndex}, n{nodeIndex}, w{weightIndex} = {gradient}");
        }

        // Calculate bias gradient
        double biasGradient = error * node.ActivationDerivative(node.Sum);
        NodeSteps[layerIndex][nodeIndex].BiasStep += biasGradient;
        Console.WriteLine($"Bias Gradient l{layerIndex}, n{nodeIndex} {biasGradient}");
    }

    private double CalculateErrorFromNextLayer(int currentLayerIndex, int currentNodeIndex)
    {
        double totalError = 0;
        var nextLayer = Layers[currentLayerIndex + 1];

        // Sum errors from all nodes in next layer that this node connects to
        for (int nextNodeIndex = 0; nextNodeIndex < nextLayer.Nodes.Length; nextNodeIndex++)
        {
            var nextNode = nextLayer.Nodes[nextNodeIndex];

            // Get the weight connecting current node to next node
            double weight = nextNode.Weights[currentNodeIndex];

            // Use the ALREADY CALCULATED error from next layer (no recursion!)
            double nextNodeError = nodeErrors[currentLayerIndex + 1][nextNodeIndex];

            // Add contribution to total error
            totalError += nextNodeError * nextNode.ActivationDerivative(nextNode.Sum) * weight;
        }

        return totalError;
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
