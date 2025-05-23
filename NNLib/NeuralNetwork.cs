using System;

namespace BackPropagation.NNLib;

public interface INeuralNetwork
{
    double[] Predict(double[] inputs);
}

class NeuralNetwork : INeuralNetwork
{
    private ILayer[] layers;
    private double learningRate = 0.01;
    private bool log = false;

    public NeuralNetwork(ILayer[] layers)
    {
        this.layers = layers;
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