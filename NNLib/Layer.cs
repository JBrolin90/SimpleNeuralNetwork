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
    public ILayer PreviousLayer { get; set; }
    public ILayer NextLayer { get; set; }
    public double[]? Inputs { get; set; }
    double[] Forward(double[] inputs);
    NodeSteps[] Backward(double dSSR);
    double GetWeightChainFactor(int index);
    double GetBiasChainFactor(int inputIndex);
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
    private ILayer? prevLayer = null!;
    public ILayer PreviousLayer
    {
        get
        {
            if (prevLayer == null)
            {
                throw new InvalidOperationException("PreviousLayer is not set. Ensure to set it before accessing.");
            }
            return prevLayer;
        }
        set
        {
            prevLayer = value ?? throw new ArgumentNullException(nameof(value), "PreviousLayer cannot be null.");
        }
    }
    private ILayer nextLayer = null!;
    public ILayer NextLayer
    {
        get
        {
            if (nextLayer == null)
            {
                throw new InvalidOperationException("NextLayer is not set. Ensure to set it before accessing.");
            }
            return nextLayer;
        }
        set
        {
            nextLayer = value ?? throw new ArgumentNullException(nameof(value), "NextLayer cannot be null.");
        }
    }
    #endregion
    #region Constructors
    public Layer(INodeFactory NodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    {
        Weights = weights;
        Biases = biases;
        Nodes = new INode[Biases.Length];
        Ys = new double[Nodes.Length];
        ActivationFunction = activationFunction;
        for (int i = 0; i < Biases.Length; i++)
        {
            Nodes[i] = NodeFactory.Create(this, i, Weights[i], Biases[i], ActivationFunction);
        }
    }
    #endregion
    #region Forward
    public virtual double[] Forward(double[] inputs)
    {
        Inputs = inputs;
        for (int i = 0; i < Nodes.Length; i++)
        {
            Ys[i] = Nodes[i].ProcessInputs(inputs);
        }
        return Ys;
    }
    #endregion
    #region Backward

    public NodeSteps[] Backward(double dSSR)
    {
        NodeSteps[] steps = new NodeSteps[Nodes.Length];
        for (int i = 0; i < Nodes.Length; i++)
        {
            steps[i] = Nodes[i].Backward(dSSR);
        }
        return steps;
    }

    public double GetWeightChainFactor(int inputIndex)
    {
        double chainFactor = NextLayer.GetWeightChainFactor(inputIndex);
        double otherChainFactor = 0;
        for (int i = 0; i < Nodes.Length; i++)
        {
            otherChainFactor += Nodes[i].GetWeightDerivativeW(i);
        }
        return chainFactor * otherChainFactor;
    }
    public double GetBiasChainFactor(int inputIndex)
    {
        double chainFactor = NextLayer.GetBiasChainFactor(inputIndex);
        double otherChainFactor = 0;
        for (int i = 0; i < Nodes.Length; i++)
        {
            otherChainFactor += Nodes[i].BiasDerivative();
        }
        return chainFactor * otherChainFactor;
    }
    #endregion

}
