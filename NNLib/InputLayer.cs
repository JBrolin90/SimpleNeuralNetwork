using System;

namespace BackPropagation.NNLib;

public class InputLayer : ILayer
{
    #region Properties
    public INode[] Nodes { get; set; } = Array.Empty<INode>();
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
    public double[]? Inputs { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
    #endregion

    public NodeSteps[] Backward(double dSSR, NodeSteps[] steps)
    {
        return steps;
    }

    public double[] Forward(double[] inputs)
    {
        return inputs; // In an input layer, the forward pass simply returns the inputs as outputs.
    }

    public double GetBiasChainFactor()
    {
        throw new NotImplementedException();
    }

    public double GetWeightChainFactor(int index)
    {
        throw new NotImplementedException();
    }
}
