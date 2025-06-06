using System;

namespace BackPropagation.NNLib;

#region Factory
public interface INodeFactory
{
    INode Create(Layer layer, int index, double[] weights, double[] bias, Func<double, double>? activationFunction = null);
}

public class NodeFactory : INodeFactory
{
    public INode Create(Layer layer, int index, double[] weights, double[] bias, Func<double, double>? activationFunction = null)
    {
        return new Node(layer, index, weights, bias, activationFunction);
    }
}

#endregion

public interface INode
{
    double[] Weights { get; set; }
    double[] Bias { get; set; }
    public double Sum { get; set; }
    Func<double, double> ActivationFunction { get; set; }
    Func<double, double> ActivationDerivative { get; set; }
    double ProcessInputs(double[] inputs);
    NodeSteps Backward(double error, NodeSteps steps);
    double GetWeightDerivativeW(int index);

    double BiasDerivative();
}

public class Node : INode
{
    #region Properties
    public ILayer Layer { get; set; } = null!; // Reference to the layer this node belongs to
    public int Index { get; set; } = -1; // Index of the node in the layer
    public double[] Weights { get; set; }
    public double[] Bias { get; set; }
    public double[] WeightDerivatives { get; set; }
    public double Sum { get; set; } = 0;
    public double Y { get; set; } = 0;
    public double[] Xs { get; set; }
    public Func<double, double> ActivationFunction { get; set; } = ActivationFunctions.SoftPlus;
    public Func<double, double> ActivationDerivative { get; set; } = ActivationFunctions.SoftPlusDerivative;
    #endregion
    #region Constructors
    public Node(ILayer layer, int index, double[] weights, double[] bias, Func<double, double>? activationFunction = null)
    {
        Layer = layer ?? throw new ArgumentNullException(nameof(layer));
        Index = index;
        Weights = weights;
        Bias = bias; // store bias by reference using array
        WeightDerivatives = new double[weights.Length];
        Xs = new double[weights.Length]; // Initialize Xs array to match weights length
        ActivationFunction = activationFunction ?? ActivationFunctions.SoftPlus;
        if (ActivationFunction == ActivationFunctions.SoftPlus)
        {
            ActivationDerivative = ActivationFunctions.SoftPlusDerivative;
        }
        else if (ActivationFunction == ActivationFunctions.Sigmoid)
        {
            ActivationDerivative = ActivationFunctions.SigmoidDerivative;
        }
        else if (ActivationFunction == ActivationFunctions.Unit)
        {
            ActivationDerivative = ActivationFunctions.UnitDerivative;
        }
    }
    #endregion
    #region Feed forward
    public double ProcessInputs(double[] xs)
    {
        Xs = xs;
        Sum = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            Sum += xs[i] * Weights[i];
        }
        Sum += Bias[0];
        Y = ActivationFunction(Sum);
        return Y;
    }
    #endregion
    #region Backpropagation

    public virtual NodeSteps Backward(double error, NodeSteps nodeSteps)
    {
        double cf = 1;
        if (Layer.NextLayer != null)
        {
            cf = Layer.NextLayer!.GetWeightChainFactor(Index);
        }

        for (int i = 0; i < Weights.Length; i++)
        {
            double wd = GetWeightDerivativeX(i);
            double wStep = error * cf * wd;
            nodeSteps.WeightSteps[i] += wStep;
        }
        double bd = BiasDerivative();
        double bStep = error * cf * bd;
        nodeSteps.BiasStep += bStep;

        return nodeSteps;
    }


    public double GetWeightDerivativeX(int index)
    {
        return Xs[index] * ActivationDerivative(Sum);
    }
    public double GetWeightDerivativeW(int index)
    {
        return Weights[index] * ActivationDerivative(Sum);
    }

    public double BiasDerivative()
    {
        return 1 * ActivationDerivative(Sum);
    }

    #endregion
}