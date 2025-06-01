using System;

namespace BackPropagation.NNLib;

#region Factory
public interface INodeFactory
{
    INode Create(double[] weights, double[] bias, Func<double, double>? activationFunction = null);
}

public class NodeFactory : INodeFactory
{
    public INode Create(double[] weights, double[] bias, Func<double, double>? activationFunction = null)
    {
        return new Node(weights, bias, activationFunction);
    }
}

#endregion

public interface INode
{
    double[] Weights { get; set; }
    double[] Bias { get; set; }
    Func<double, double> ActivationFunction { get; set; }
    Func<double, double> ActivationDerivative { get; set; }
    NodeSteps Backward(NodeSteps nodeSteps, double error, Func<double> GetChainFactor);
    double GetChainFactor();
    double ProcessInputs(double[] inputs);
}

public class Node : INode
{
    #region Properties
    public double[] Weights { get; set; }
    public double[] Bias { get; set; }
    public double[] WeightDerivatives { get; set; }
    public double Sum { get; set; } = 0;
    public double Y { get; set; } = 0;
    public double[]? Xs { get; set; }
    public Func<double, double> ActivationFunction { get; set; } = ActivationFunctions.SoftPlus;
    public Func<double, double> ActivationDerivative { get; set; } = ActivationFunctions.SoftPlusDerivative;
    #endregion
    #region Constructors
    public Node(double[] weights, double[] bias, Func<double, double>? activationFunction = null)
    {
        Weights = weights;
        Bias = bias; // store bias by reference using array
        WeightDerivatives = new double[weights.Length];
        ActivationFunction = activationFunction ?? ActivationFunctions.SoftPlus;
        if (ActivationFunction == ActivationFunctions.SoftPlus)
        {
            ActivationDerivative = ActivationFunctions.SoftPlusDerivative;
        }
        else if (ActivationFunction == ActivationFunctions.Sigmoid)
        {
            ActivationDerivative = ActivationFunctions.SigmoidDerivative;
        }
        else if (ActivationDerivative == ActivationFunctions.Unit)
        {
            ActivationFunction = ActivationFunctions.UnitDerivative;
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
        Y = (ActivationFunction ?? ActivationFunctions.Unit)(Sum);
        return Y;
    }
    #endregion
    #region Backpropagation

    public NodeSteps Backward(NodeSteps nodeSteps, double error, Func<double> GetChainFactor)
    {
        if (Xs == null)
        {
            throw new InvalidOperationException("Inputs must be processed before backpropagation.");
        }

        nodeSteps.WeightSteps = new double[Weights.Length];
        for (int i = 0; i < Weights.Length; i++)
        {
            nodeSteps.WeightSteps[i] = GetFullWeightStep(i, error, GetChainFactor) * error;
        }
        nodeSteps.BiasStep = BiasDerivative() * error;

        return nodeSteps;
    }
    public double GetChainFactor()
    {
        double x = 0;
        x += GetWeightDerivativeX(i) * ActivationDerivative(Sum);
        return x;

    }

    public double GetFullWeightStep(int index, double error, Func<double> GetChainFactor)
    {
        double[] fullWeightsStep = new double[Weights.Length];
        if (index < 0 || index >= Weights.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the range of weights.");
        }
        fullWeightStep = GetWeightDerivativeW(index) * GetChainFactor() * error;
        return fullWeightStep;
    }

    public double FullBiasStep(double error, Func<double> GetChainFactor)
    {
        double fullBiasStep = BiasDerivative() * GetChainFactor() * error;
        return fullBiasStep;
    }


    private double GetWeightDerivativeW(int index)
    {
        if (index < 0 || index >= Weights.Length || Xs == null)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the range of weights and inputs must be processed.");
        }
        return Xs[index] * ActivationDerivative(Sum);
    }
    private double GetWeightDerivativeX(int index)
    {
        if (index < 0 || index >= Weights.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the range of weights.");
        }
        return Weights[index] * ActivationDerivative(Sum);
    }

    public double BiasDerivative()
    {
        return 1;
    }

    #endregion
}