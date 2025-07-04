using System;
using System.ComponentModel;
using System.Globalization;
using System.Net.WebSockets;
using System.Runtime.Intrinsics.Arm;

namespace BackPropagation.NNLib;


public class Gradients
{
    public double[] WeightGradient;
    public double BiasGradient = 0;

    public Gradients(int weightCount)
    {
        WeightGradient = new double[weightCount];
        for (int i = 0; i < weightCount; i++)
        {
            WeightGradient[i] = 0;
        }
    }
}

public class NeuralNetworkTrainer : NeuralNetwork
{

    public Func<double[][], double[][], double[]> LossFunction = LossFunctions.SumSquaredError;
    public Func<double[][], double[][], double[]> LossFunctionD = LossFunctions.SumSquaredErrorDerivative;
    public double LearningRate = 0;
    public Gradients[][] Gradients = [];

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
    public double[] TrainOneEpoch(double[][] trainingData, double[][] observed)
    {
        int outputCount = Layers[^1].Nodes.Length;

        double[][] predictions = new double[trainingData.Length][];
        PrepareBackPropagation();
        ResetGradients();
        loss = new double[observed[0].Length];
        dLoss = new double[observed[0].Length];
        for (int i = 0; i < trainingData.Length; i++)
        {
            // Forward pass
            predictions[i] = Predict(trainingData[i]);
            double[] lossPart = LossFunctions.SquaredError(predictions[i], observed[i]);
            double[] dLossPart = LossFunctions.SquaredErrorDerivative(predictions[i], observed[i]);
            int count = predictions[i].Length;
            for (int j = 0; j < count; j++)
            {
                loss[j] += lossPart[j];
                dLoss[j] += dLossPart[j];
            }
            PropagateBackwards(dLossPart);
            var node = Layers[0].Nodes[0];
            var wGrd = Gradients[0][0].WeightGradient[0];
            var bGrd = Gradients[0][0].BiasGradient;
            var w = Weigths[0][0][0];
            var b = Biases[0][0][0];
            // Console.WriteLine($"o:{observed[i][0]}, i:{trainingData[i][0]}, w{w}, b{b}, XY{node.Sum} L{lossPart[0]}, dL{dLossPart[0]}, wGrd{wGrd}, bGrd{bGrd} ");
        }
        UpdateWeightsAndBiases(trainingData.Length);

        return loss;
    }
    public double[] TrainOneEpoch(Sample[] trainingData)
    {
        double[][] predictions = new double[trainingData.Length][];
        PrepareBackPropagation();
        ResetGradients();
        loss = new double[trainingData.Length];
        dLoss = new double[trainingData.Length];
        for (int i = 0; i < trainingData.Length; i++)
        {
            // Forward pass
            predictions[i] = Predict(trainingData[i].Xample);
            double[] lossPart = LossFunctions.SquaredError(predictions[i], [trainingData[i].Observed]);
            double[] dLossPart = LossFunctions.SquaredErrorDerivative(predictions[i], [trainingData[i].Observed]);
            int count = predictions[i].Length;
            for (int j = 0; j < count; j++)
            {
                loss[j] += lossPart[j];
                dLoss[j] += dLossPart[j];
            }
            PropagateBackwards(dLossPart);
        }
        UpdateWeightsAndBiases(trainingData.Length);

        return loss;
    }



    public void PrepareBackPropagation()
    {
        Gradients = new Gradients[Layers.Length][];
        int i = 0;
        foreach (var layer in Layers)
        {
            Gradients[i] = new Gradients[layer.Nodes.Length];
            for (int j = 0; j < layer.Nodes.Length; j++)
            {
                Gradients[i][j] = new Gradients(layer.Nodes[j].Weights.Length);
            }
            i++;
        }
    }

    private void ResetGradients()
    {
        for (int i = 0; i < Gradients.Length; i++)
        {
            for (int j = 0; j < Gradients[i].Length; j++)
            {
                Gradients[i][j].BiasGradient = 0;
                for (int k = 0; k < Gradients[i][j].WeightGradient.Length; k++)
                {
                    Gradients[i][j].WeightGradient[k] = 0;
                }
            }
        }
    }


    public void BackPropagateRecursive(double[] dSSR)
    {
        int i = 0;
        foreach (var layer in Layers)
        {
            layer.Backward(dSSR[0], Gradients[i]);
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
        double[] layerInputs = layer.Inputs ?? [];

        // Calculate weight gradients
        double activationDerivative = node.ActivationDerivative(node.Sum);
        for (int weightIndex = 0; weightIndex < node.Weights.Length; weightIndex++)
        {
            double input = (weightIndex < layerInputs.Length) ? layerInputs[weightIndex] : 0;
            double gradient = error * activationDerivative * input;
            Gradients[layerIndex][nodeIndex].WeightGradient[weightIndex] += gradient;
        //    Console.WriteLine($"Weight Gradient [l{layerIndex}, n{nodeIndex}, w{weightIndex}] = isol: {gradient} Sum: {Gradients[layerIndex][nodeIndex].WeightGradient[weightIndex]}");
        }

        // Calculate bias gradient
        double biasGradient = error * node.ActivationDerivative(node.Sum) * 1;
        Gradients[layerIndex][nodeIndex].BiasGradient += biasGradient;
        //Console.WriteLine($"Bias Gradient [l{layerIndex}, n{nodeIndex}] = isol: {biasGradient} Sum {Gradients[layerIndex][nodeIndex].BiasGradient}");
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
            // dP/dY
            double weight = nextNode.Weights[currentNodeIndex];

            // dY/dX
            double activationDerivative = nextNode.ActivationDerivative(nextNode.Sum);

            // Use the ALREADY CALCULATED error from next layer (no recursion!)
            double nextNodeError = nodeErrors[currentLayerIndex + 1][nextNodeIndex];


            // Add contribution to total error
            totalError += activationDerivative * weight * nextNodeError;
        }

        return totalError;
    }

    private void UpdateWeightsAndBiases(int divisor = 1)
    {
        // Update weights and biases
        for (int j1 = 0; j1 < Weigths.Length; j1++)
        {
            for (int k1 = 0; k1 < Weigths[j1].Length; k1++)
            {
                for (int l = 0; l < Weigths[j1][k1].Length; l++)
                {
                    double wGradient = Gradients[j1][k1].WeightGradient[l];
                    Weigths[j1][k1][l] -= wGradient * LearningRate / divisor; 
                }
                double bGradient = Gradients[j1][k1].BiasGradient;
                Biases[j1][k1][0] -= bGradient * LearningRate /divisor;
            }
        }
    }
}
