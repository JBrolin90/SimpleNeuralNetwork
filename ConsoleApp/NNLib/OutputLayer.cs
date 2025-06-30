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
    public OutputLayer(int index, INeuronFactory NodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    : base(index, NodeFactory, weights, biases, activationFunction)
    {
    }
    #endregion

}
