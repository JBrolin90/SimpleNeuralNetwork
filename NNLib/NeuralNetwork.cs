using System;

namespace BackPropagation.NNLib;

public interface INeuralNetwork
{
    double[] Predict(double[] inputs);
}

public class NeuralNetwork : INeuralNetwork
{
    private ILayer[] layers;
    private double learningRate = 0.01;
    private bool log = false;

    public NeuralNetwork(ILayerFactory LayerFactory, INodeFactory NodeFactory,
                        double[][][] weights, double[][] biases, Func<double, double>[]? activationFunctions = null,
                        double learningRate = 0.01)
    {
        this.learningRate = learningRate;
        layers = new ILayer[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            layers[i] = LayerFactory.Create(NodeFactory, weights[i], biases[i], activationFunctions?[i] ?? Node.SoftPlus);
        }
    }
    public double[] Predict(double[] inputs)
    {
        double[] outputs = inputs;
        foreach (var layer in layers)
        {
            outputs = layer.Forward(outputs);
        }
        return outputs;
    }

}