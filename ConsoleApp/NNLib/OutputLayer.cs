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

    public override double CalculateLayerErrorRecursively(int inputIndex)
    {
        double thisNodeTotalError = 0;
        for (int nodeIndex = 0; nodeIndex < Nodes.Length; nodeIndex++)
        {
            var node = Nodes[nodeIndex];
            var weight = node.Weights[inputIndex];
            var activationDerivative = node.ActivationDerivative(node.Sum);
            thisNodeTotalError +=  activationDerivative * weight;
        }
        return thisNodeTotalError;
    }


    public override double GetBiasChainFactor()
    {
        return 1;
    }

}
