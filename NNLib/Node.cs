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
    public double[] GetWeightUpdates(double[] inputs, double error);
    double Y { get; set; }

    double ProcessInputs(double[] inputs);
    double[] Descent(double dSSR);
}

public class Node : INode
{
    #region Properties
    public double[] Weights { get; set; }
    public double[] Bias { get; set; }
    public double[] WeightDerivatives { get; set; }
    double[]? WeightsResiduals { get; set; }
    public double Sum { get; set; } = 0;
    public double Y { get; set; } = 0;
    public double[]? Xs { get; set; }
    public Func<double, double> ActivationFunction { get; set; } = SoftPlus;
    public Func<double, double> ActivationDerivative { get; set; } = UnitActivationDerivative;
    #endregion
    #region Constructors
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
            ActivationFunction = UnitActivationDerivative;
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
        Y = (ActivationFunction ?? UnitActivation)(Sum);
        return Y;
    }
    #endregion
    #region Backpropagation
    public double[] GetWeightUpdates(double[] inputs, double error)
    {
        const double LearningRate = 0.01; // Set a default learning rate
        double[] weightUpdates = new double[Weights.Length];
        for (int k = 0; k < Weights.Length; k++)
        {
            weightUpdates[k] = LearningRate * GetWeightDerivativeW(k, Xs ?? throw new NullReferenceException()) - error;
        }
        return weightUpdates;
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

    public double GetWeightDerivativeW(int index, double[] xs)
    {
        if (index < 0 || index >= Weights.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the range of weights.");
        }
        return xs[index] * ActivationDerivative(xs[index] * Weights[index] + Bias[0]);
    }
    public double GetWeightDerivativeX(int index, double[] xs)
    {
        if (index < 0 || index >= Weights.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index must be within the range of weights.");
        }
        return Weights[index] * ActivationDerivative(xs[index] * Weights[index] + Bias[0]);
    }

    public double BiasDerivative()
    {
        return 1;
    }
    #endregion
    #region Static Activation Functions and derivatives
    public static double UnitActivation(double x) => x;
    public static double UnitActivationDerivative(double x) => 1;

    public static double SoftPlus(double x)
    {
        return Math.Log(1 + Math.Exp(x));
    }
    public static double SoftPlusDerivative(double x)
    {
        return Sigmoid(x);
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
