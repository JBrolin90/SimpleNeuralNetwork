using System;

namespace BackPropagation.NNLib;

public interface INeuralNetwork
{
    double[] Predict(double[] inputs);
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
        Layers = new ILayer[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            Layers[i] = LayerFactory.Create(NodeFactory, weights[i], biases[i], activationFunctions?[i] ?? Node.SoftPlus);
        }
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

}