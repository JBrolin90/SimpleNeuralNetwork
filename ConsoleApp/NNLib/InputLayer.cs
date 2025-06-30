using System;

namespace BackPropagation.NNLib;

public class InputLayer : Layer
{
    public override ILayer? PreviousLayer
    {
        get
        {
            return null;
        }
        set
        {
            throw new InvalidOperationException("Input layer cannot have a previous layer");
        }
    }


    public InputLayer(int index, INeuronFactory NodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    : base(index, NodeFactory, weights, biases, activationFunction)
    {
    }


}
