using System;

namespace BackPropagation.NNLib;

public interface INeuralNetwork
{
    public ILayer[] Layers { get; set; }
    public double[][] Ys { get; set; }
    public double[][][] Weigths { get; set; }
    public double[][][] Biases { get; set; }

    double[] Predict(double[] inputs);
    public double[][][] BackwardPass(double[] inputs, double[] errors);
}

public class NeuralNetwork : INeuralNetwork
{
    public ILayer[] Layers { get; set; }
    public double[][] Ys { get; set; }
    public double[][][] Weigths { get; set; }
    public double[][][] Biases { get; set; }

    public NeuralNetwork(ILayerFactory LayerFactory, INodeFactory NodeFactory,
                        double[][][] weights, double[][][] biases, double[][] ys, Func<double, double>[]? activationFunctions = null,
                        double learningRate = 0.01)
    {
        Ys = ys;
        Weigths = weights;
        Biases = biases;
        Layers = new ILayer[weights.Length + 2]; // +2 for the input and output layers

        for (int i = 0; i < Layers.Length; i++)
        {
            var layerType = i == 0 ? LayerType.Input : i == Layers.Length - 1 ? LayerType.Output : LayerType.Hidden;
            Layers[i] = LayerFactory.Create(NodeFactory, weights[i], biases[i], activationFunctions?[i] ?? Node.SoftPlus, layerType);
            if (i > 0)
            {
                Layers[i].PreviousLayer = Layers[i - 1];
                Layers[i - 1].NextLayer = Layers[i];
            }
        }
        Layers[weights.Length] = LayerFactory.Create(NodeFactory, weights[^1], biases[^1], activationFunctions?[^1] ?? Node.SoftPlus, LayerType.Output);
    }
    public double[] Predict(double[] inputs)
    {
        int i = 0;
        double[] outputs = inputs;
        foreach (var layer in Layers)
        {
            outputs = layer.Forward(outputs);
            Ys[i++] = outputs;
        }
        return outputs;
    }

    public double[][][] BackwardPass(double[] inputs, double[] errors)
    {
        double[][][] weightUpdates = new double[Layers.Length][][];
        for (int i = Layers.Length - 1; i >= 0; i--)
        {
            var layer = Layers[i];
            weightUpdates[i] = layer.GetWeightUpdates(inputs, errors);
        }
        return weightUpdates;
    }

}