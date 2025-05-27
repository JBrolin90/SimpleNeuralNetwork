using System;

namespace BackPropagation.NNLib;

#region Factory
public interface ILayerFactory
{
    ILayer Create(INodeFactory factory, double[][] weights, double[][] biases, Func<double, double>? activationFunctions = null);
}

public class LayerFactory : ILayerFactory
{
    public ILayer Create(INodeFactory factory, double[][] weights, double[][] biases, Func<double, double>? activationFunctions = null)
    {
        return new Layer(factory, weights, biases, activationFunctions);
    }
}
#endregion

public interface ILayer
{
    public INode[] Nodes { get; set; }
    double[] Forward(double[] inputs);
    // public double[][] Backward();
    double[][] Descent(double dSSR);

    static double UnitActivation(double x) => x;
    static double SoftPlus(double x) => Math.Log(1 + Math.Exp(x));
}


public class Layer : ILayer
{
    public INode[] Nodes { get; set; }
    public double[]? Inputs { get; set; }
    public double[][] Weights { get; set; }
    public double[][] Biases { get; set; }
    public double[] Ys { get; set; }

    Func<double, double>? ActivationFunction { get; set; }
    public Layer(INodeFactory NodeFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    {
        Weights = weights;
        Biases = biases;
        Nodes = new INode[Biases.Length];
        Ys = new double[Nodes.Length];
        ActivationFunction = activationFunction;
        for (int i = 0; i < Biases.Length; i++)
        {
            Nodes[i] = NodeFactory.Create(Weights[i], Biases[i], ActivationFunction);
        }
    }

    public double[] Forward(double[] inputs)
    {
        Inputs = inputs;
        for (int i = 0; i < Nodes.Length; i++)
        {
            Ys[i] = Nodes[i].ProcessInputs(inputs);
        }
        return Ys;
    }

    public double[][] Descent(double dSSR)
    {
        double[][] gradients = new double[Nodes.Length][];
        for (int i = 0; i < Nodes.Length; i++)
        {
            gradients[i] = Nodes[i].Descent(dSSR);
            // var biasDerivative = Nodes[i].BiasDerivative();
            // Biases[i][0] -= biasDerivative * 0.01; // Update the bias for the node
        }
        return gradients;
    }

    // public double[][] Backward()
    // {
    //     // Implement the backward pass logic here
    //     // This is a placeholder implementation
    //     double[][] gradients = new double[Nodes.Length][];
    //     int i = 0;
    //     foreach (INode node in Nodes)
    //     {
    //         gradients[i++] = node.WeightDerivatives(Inputs);
    //         //var biasDerivative = node.BiasDerivative();
    //     }
    //     return gradients;
    // }


}
