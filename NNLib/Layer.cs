using System;

namespace BackPropagation.NNLib;

public interface ILayer
{
    double[] Forward(double[] inputs);
    static double UnitActivation(double x) => x;
    static double SoftPlus(double x)
    {
        return Math.Log(1 + Math.Exp(x));
    }
}


public class Layer : ILayer
{
    private INode[] nodes;

    public Layer(INode[] nodes)
    {
        this.nodes = nodes;
    }

    public double[] Forward(double[] inputs)
    {
        double[] outputs = new double[nodes.Length];
        for (int i = 0; i < nodes.Length; i++)
        {
            outputs[i] = nodes[i].ProcessInputs(inputs);
        }
        return outputs;
    }
}
