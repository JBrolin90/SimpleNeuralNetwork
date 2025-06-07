using System;

namespace BackPropagation.NNLib;

public class OutputLayer : Layer
{

    public override ILayer? NextLayer
    {
        get
        {
            return null;
        }
        set
        {
            throw new InvalidOperationException( "The outputlayer cannot have a next layer");
        }
    }


    #region Constructors
    public OutputLayer(int index, INodeFactory NodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    : base(index, NodeFactory, weights, biases, activationFunction)
    {
    }
    #endregion

    public override double GetWeightChainFactor(int inputIndex)
    {
        double chainFactor = 1;  
        double otherChainFactor = 0;
        for (int nodeIndex = 0; nodeIndex < Nodes.Length; nodeIndex++)
        {
            otherChainFactor += Nodes[nodeIndex].GetWeightDerivativeW(inputIndex);
        }
        return chainFactor * otherChainFactor;
    }


    public override double GetBiasChainFactor()
    {
        return 1;
    }

}
