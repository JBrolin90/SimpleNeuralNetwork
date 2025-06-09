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
    public int Index { get; set; }
    public INode[] Nodes { get; set; }
    public ILayer? PreviousLayer { get; set; }
    public ILayer? NextLayer { get; set; }
    public double[]? Inputs { get; set; }
    double[] Forward(double[] inputs);
    NodeSteps[] Backward(double dSSR, NodeSteps[] steps);
    double CalculateLayerErrorRecursively(int index);
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
    protected ILayer? nextLayer = null;
    public virtual ILayer? NextLayer
    {
        get
        {
            return nextLayer;
        }
        set
        {
            nextLayer = value;
        }
    }
    
    protected ILayer? prevLayer = null;
    public virtual ILayer? PreviousLayer
    {
        get
        {
            return prevLayer;
        }
        set
        {
            prevLayer = value;
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

    public virtual NodeSteps[] Backward(double dSSR, NodeSteps[] nodeSteps)
    {
        for (int nodeIndex = 0; nodeIndex < Nodes.Length; nodeIndex++)
        {
            nodeSteps[nodeIndex] = Nodes[nodeIndex].Backward(dSSR, nodeSteps[nodeIndex]);
        }
        return nodeSteps;
    }

    //Calculate Layer Error Recursively
    public virtual double CalculateLayerErrorRecursively(int inputIndex)
    {
        if (NextLayer == null)
        {
            return 1.0; // Terminal layer, no chain factor multiplication needed
        }

        double nextLayerError = NextLayer.CalculateLayerErrorRecursively(inputIndex);
        double thisNodeTotalError = 0;
        for (int nodeIndex = 0; nodeIndex < Nodes.Length; nodeIndex++)
        {
            var node = Nodes[nodeIndex];
            var weight = node.Weights[inputIndex];
            var activationDerivative = node.ActivationDerivative(node.Sum);
            thisNodeTotalError += nextLayerError * activationDerivative * weight;
        }
        return thisNodeTotalError;
    }
    
    public virtual double GetBiasChainFactor()
    {
        if (NextLayer == null)
        {
            return 1.0; // Terminal layer, no chain factor multiplication needed
        }
        
        double chainFactor = NextLayer.GetBiasChainFactor();
        double otherChainFactor = 0;
        for (int nodeIndex = 0; nodeIndex < Nodes.Length; nodeIndex++)
        {
            otherChainFactor += Nodes[nodeIndex].BiasDerivative();
        }
        return chainFactor * otherChainFactor;
    }
    #endregion
}
