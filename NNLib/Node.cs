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
    double BiasDerivative();
    public double[] GetWeightsResiduals(double dSSR, double[] xs);
    double Y { get; set; }

    double ProcessInputs(double[] inputs);
    double[] Descent(double dSSR);
}

public class Node : INode
{
    public double[] Weights { get; set; }
    public double[] Bias { get; set; }
    public double[] WeightDerivatives { get; set; }
    double[]? WeightsResiduals { get; set; }
    public double Sum { get; set; } = 0;
    public double Y { get; set; } = 0;
    public double[]? Xs { get; set; }
    public Func<double, double> ActivationFunction { get; set; } = SoftPlus;
    public Func<double, double> ActivationDerivative { get; set; } = UnitActivation;

    public Node(double[] weights, double[] bias, Func<double, double>? activationFunction = null)
    {
        Weights = weights;
        Bias = bias; // store bias by reference using array
        WeightDerivatives = new double[weights.Length];
        ActivationFunction = activationFunction ?? SoftPlus;
        if (ActivationFunction == SoftPlus)
        {
            ActivationDerivative = SoftPlusDerivative;
        }
        else if (ActivationFunction == Sigmoid)
        {
            ActivationDerivative = SigmoidDerivative;
        }
        else if (ActivationDerivative == UnitActivation)
        {
            ActivationFunction = UnitActivation;
        }
    }

    public double ProcessInputs(double[] xs)
    {
        Xs = xs;
        Sum = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            Sum += xs[i] * Weights[i];
        }
        Sum += Bias[0];
        Y = (ActivationFunction ?? UnitActivation)(Sum);
        return Y;
    }

    public double[] Descent(double dSSR)
    {
        if (Xs == null)
        {
            throw new InvalidOperationException("Inputs must be processed before descent.");
        }

        WeightsResiduals = GetWeightsResiduals(dSSR, Xs);
        return WeightsResiduals;
    }


    public double[] GetWeightsResiduals(double dSSR, double[] xs)
    {
        WeightsResiduals = new double[Weights.Length];
        for (int i = 0; i < Weights.Length; i++)
        {
            if (i == 0)
                WeightDerivatives[i] = ActivationDerivative(xs[i]);
            else
                WeightDerivatives[i] = ActivationDerivative(xs[i]);
            WeightsResiduals[i] = WeightDerivatives[i] * dSSR;
        }
        return WeightsResiduals;
    }

    public double WeightDerivative(int index, double[] xs)
    {
        if (index < 0 || index >= Weights.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the range of weights.");
        }
        return xs[index] * ActivationDerivative(Sum);
    }
    public double WeightDerivative(int index)
    {
        if (index < 0 || index >= Weights.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the range of weights.");
        }
        return Weights[index] * ActivationDerivative(Sum);
    }
    public double BiasDerivative()
    {
        return ActivationDerivative(Sum);
    }



    #region Static Activation Functions
    public static double UnitActivation(double x) => x;

    public static double SoftPlus(double x)
    {
        return Math.Log(1 + Math.Exp(x));
    }
    public static double SoftPlusDerivative(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
    public static double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
    public static double SigmoidDerivative(double x)
    {
        return x * (1 - x);
    }

    #endregion
}
