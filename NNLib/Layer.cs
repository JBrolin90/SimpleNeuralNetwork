using System;

namespace BackPropagation.NNLib;

#region Factory

public enum LayerType
{
    Input,
    Hidden,
    Output
}
public interface ILayerFactory
{
    ILayer Create(INodeFactory factory, double[][] weights, double[][] biases, Func<double, double>? activationFunctions = null, LayerType layerType = LayerType.Hidden);
}

public class LayerFactory : ILayerFactory
{
    public ILayer Create(INodeFactory factory, double[][] weights, double[][] biases, Func<double, double>? activationFunctions = null, LayerType layerType = LayerType.Hidden)
    {
        if (layerType == LayerType.Input)
        {
            throw new NotImplementedException("Input layer creation is not yet implemented. Use a different layer type.");
        }
        if (layerType == LayerType.Output)
        {
            return new OutputLayer(factory, weights, biases, activationFunctions);
        }
        return new Layer(factory, weights, biases, activationFunctions);
    }
}
#endregion
#region interfaces
public interface ILayer
{
    public INode[] Nodes { get; set; }
    public ILayer? PreviousLayer { get; set; }
    public ILayer? NextLayer { get; set; }
    public double[]? Inputs { get; set; }
    double[] Forward(double[] inputs);
    // public double[][] Backward();
    double[][] GetWeightUpdates(double[] inputs, double[] errors);

    double[][] Descent(double dSSR);

    static double UnitActivation(double x) => x;
    static double SoftPlus(double x) => Math.Log(1 + Math.Exp(x));
}
#endregion


public class Layer : ILayer
{
    #region Properties
    public INode[] Nodes { get; set; }
    public double[]? Inputs { get; set; }
    public double[][] Weights { get; set; }
    public double[][] Biases { get; set; }
    public double[] Ys { get; set; }
    Func<double, double>? ActivationFunction { get; set; }
    public ILayer? PreviousLayer { get; set; } = null;
    public ILayer? NextLayer { get; set; } = null;
    #endregion
    public Layer(INodeFactory NodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    {
        Weights = weights;
        Biases = biases;
        Nodes = new INode[Biases.Length];
        Ys = new double[Nodes.Length];
        ActivationFunction = activationFunction;
        for (int i = 0; i < Biases.Length; i++)
        {
            Nodes[i] = NodeFactory.Create(Weights[i], Biases[i], ActivationFunction);
        }
    }

    public virtual double[] Forward(double[] inputs)
    {
        Inputs = inputs;
        for (int i = 0; i < Nodes.Length; i++)
        {
            Ys[i] = Nodes[i].ProcessInputs(inputs);
        }
        return Ys;
    }

    public virtual double[][] GetWeightUpdates(double[] inputs, double[] errors)
    {
        double[][] weightUpdates = new double[Nodes.Length][];
        for (int j = 0; j < Nodes.Length; j++)
        {
            weightUpdates[j] = Nodes[j].GetWeightUpdates(inputs, errors[j]);
        }
        return weightUpdates;
    }


    public double[][] Descent(double dSSR)
    {
        double[][] gradients = new double[Nodes.Length][];
        for (int i = 0; i < Nodes.Length; i++)
        {
            gradients[i] = Nodes[i].Descent(dSSR);
        }
        return gradients;
    }

}
