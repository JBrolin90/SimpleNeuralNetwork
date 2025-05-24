using System;

namespace BackPropagation.NNLib;

public interface ILayer
{
    double[] Forward(double[] inputs);

    static double UnitActivation(double x) => x;
    static double SoftPlus(double x) => Math.Log(1 + Math.Exp(x));
}

public interface ILayerFactory
{
    ILayer Create(INodeFactory factory, double[][] weights, double[] biases, Func<double, double>? activationFunctions = null);
}

public class LayerFactory : ILayerFactory
{
    public ILayer Create(INodeFactory factory, double[][] weights, double[] biases, Func<double, double>? activationFunctions = null)
    {
        return new Layer(factory, weights, biases, activationFunctions);
    }
}

public class Layer : ILayer
{
    private readonly INode[] nodes;
    private readonly double[][] weights;
    public Layer(INodeFactory NodeFactory, double[][] weights, double[] biases, Func<double, double>? activationFunction = null)
    {
        this.weights = weights;
        nodes = new INode[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            nodes[i] = NodeFactory.Create(weights[i], biases[i], activationFunction);
        }
    }


    public double[] Forward(double[] inputs)
    {
        double[] outputs = new double[nodes.Length];
        for (int i = 0; i < nodes.Length; i++)
        {
            outputs[i] = nodes[i].ProcessInputs(inputs);
        }
        return outputs;
    }
}
