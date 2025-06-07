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
    ILayer Create(int index, INodeFactory factory, double[][] weights, double[][] biases, Func<double, double>? activationFunctions = null, LayerType layerType = LayerType.Hidden, double[]?  expectedOutputs = null);
}

public class LayerFactory : ILayerFactory
{
    public ILayer Create(int index, INodeFactory factory, double[][] weights, double[][] biases, Func<double, double>? activationFunctions = null, LayerType layerType = LayerType.Hidden, double[]? expectedOutputs = null)
    {
        if (layerType == LayerType.Input)
        {
            return new InputLayer(index, factory, weights, biases, activationFunctions);
        }
        if (layerType == LayerType.Output)
        {
            return new OutputLayer(index, factory, weights, biases, activationFunctions);
        }
        return new Layer(index, factory, weights, biases, activationFunctions);
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
    NodeSteps[] Backward(double dSSR, NodeSteps[] steps);
    double GetWeightChainFactor(int index);
    double GetBiasChainFactor();
}
#endregion


public class Layer : ILayer
{
    #region Properties
    public int Index { get; set; } = -1; // Index of the layer in the network
    public INode[] Nodes { get; set; }
    public double[]? Inputs { get; set; }
    public double[][] Weights { get; set; }
    public double[][] Biases { get; set; }
    public double[] Ys { get; set; }
    Func<double, double>? ActivationFunction { get; set; }
    protected ILayer? prevLayer = null!;
    public virtual ILayer? PreviousLayer
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
    protected ILayer? nextLayer = null!;
    public virtual ILayer? NextLayer
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
    public Layer(int index, INodeFactory NodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    {
        Index = index;
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
        double[] outputs = new double[Nodes.Length];
        Ys = new double[Nodes.Length];
        for (int i = 0; i < Nodes.Length; i++)
        {
            outputs[i] = Nodes[i].ProcessInputs(Inputs);
            Ys[i] = outputs[i];
        }
        return outputs;
    }
    #endregion
    #region Backward

    public virtual NodeSteps[] Backward(double dSSR, NodeSteps[] steps)
    {
        for (int i = 0; i < Nodes.Length; i++)
        {
            steps[i] = Nodes[i].Backward(dSSR, steps[i]);
        }
        return steps;
    }

    public virtual double GetWeightChainFactor(int inputIndex)
    {
        double chainFactor = NextLayer!.GetWeightChainFactor(inputIndex);
        double otherChainFactor = 0;
        for (int nodeIndex = 0; nodeIndex < Nodes.Length; nodeIndex++)
        {
            otherChainFactor += Nodes[nodeIndex].GetWeightDerivativeW(inputIndex);
        }
        return chainFactor * otherChainFactor;
    }
    public virtual double GetBiasChainFactor()
    {
        double chainFactor = NextLayer!.GetBiasChainFactor();
        double otherChainFactor = 0;
        for (int nodeIndex = 0; nodeIndex < Nodes.Length; nodeIndex++)
        {
            otherChainFactor += Nodes[nodeIndex].BiasDerivative();
        }
        return chainFactor * otherChainFactor;
    }
    #endregion
}
