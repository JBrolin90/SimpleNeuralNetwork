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
    public OutputLayer(INodeFactory NodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    {
        Ys = new double[biases.Length];
    }
    #endregion

    public double[] Forward(double[] inputs)
    {
        Inputs = inputs;
        Ys = new double[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            Ys[i] = inputs.Sum();  //Nodes[i].ProcessInputs(inputs);
        }
        return Ys;
    }
    public NodeSteps[] Backward(double dSSR)
    {
        NodeSteps[] steps = new NodeSteps[Nodes.Length];
        return steps;
    }

    public double GetWeightChainFactor(int _)
    {
        return 1;
    }
    public double GetBiasChainFactor(int _)
    {
        return 1;
    }

}
