using System;

namespace BackPropagation.NNLib;

public class OutputLayer : ILayer
{
    public OutputLayer(INodeFactory nodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    {
    }

    public INode[] Nodes { get; set; } = Array.Empty<INode>();
    public ILayer? PreviousLayer { get; set; } = null;
    public ILayer? NextLayer { get; set; }
    public double[]? Inputs { get; set; }
    public double[][] Descent(double dSSR)
    {
        throw new NotImplementedException();
    }

    public double[] Forward(double[] inputs)
    {
        throw new NotImplementedException();
    }

    public double[][] GetWeightUpdates(double[] inputs, double[] errors)
    {
        throw new NotImplementedException();
    }
}
