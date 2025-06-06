using System;

namespace BackPropagation.NNLib;

public class OutputLayer : ILayer
{
    #region Properties
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
    public INode[] Nodes { get; set; } = Array.Empty<INode>();
    public double[] Ys { get; set; }
    public double[]? Inputs { get; set; }
    #endregion
    #region Constructors
    public OutputLayer(INodeFactory NodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null, int outputSize = 1)
    {
        Nodes = new INode[0]; // Output layer has no nodes
        Ys = new double[outputSize];
    }
    #endregion

    public double[] Forward(double[] inputs)
    {
        Inputs = inputs;
        // Since output layer has no weights/biases, we need to transform inputs to outputs
        // This is a simplified approach - in a real network, the last hidden layer would be the actual output
        // For now, we'll take the first N inputs where N is the expected output size
        for (int i = 0; i < Ys.Length && i < inputs.Length; i++)
        {
            Ys[i] = inputs[i];
        }
        // If we need more outputs than inputs, replicate the last input
        for (int i = inputs.Length; i < Ys.Length; i++)
        {
            Ys[i] = inputs.Length > 0 ? inputs[inputs.Length - 1] : 0;
        }
        return Ys;
    }
    public NodeSteps[] Backward(double dSSR, NodeSteps[] steps)
    {
        return steps;
    }

    public double GetWeightChainFactor(int _)
    {
        return 1;
    }
    public double GetBiasChainFactor()
    {
        return 1;
    }

}
