using System;

namespace BackPropagation.NNLib;

public class InputLayer : ILayer
{
    public INode[] Nodes { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
    public ILayer PreviousLayer { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
    public ILayer NextLayer { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
    public double[]? Inputs { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

    public NodeSteps[] Backward(double dSSR)
    {
        throw new NotImplementedException();
    }

    public double[] Forward(double[] inputs)
    {
        throw new NotImplementedException();
    }

    public double GetBiasChainFactor(int inputIndex)
    {
        throw new NotImplementedException();
    }

    public double GetWeightChainFactor(int index)
    {
        throw new NotImplementedException();
    }
}
