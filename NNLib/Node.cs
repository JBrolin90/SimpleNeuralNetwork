using System;

namespace BackPropagation.NNLib;

public interface INode
{
    double[] Weights { get; set; }
    double[] Bias { get; set; }
    Func<double, double> ActivationFunction { get; set; }
    Func<double, double> ActivationDerivative { get; set; }
    double Output { get; set; }

    double ProcessInputs(double[] inputs);
}

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
public class Node : INode
{
    public double[] Weights { get; set; }

    //A property to access individual weights using [] notation
    // This is a workaround for the fact that C# does not support indexers in classes
    public double this[int index]
    {
        get { return Weights[index]; }
        set { Weights[index] = value; }
    }

    public double[] Bias { get; set; }
    public Func<double, double> ActivationFunction { get; set; } = SoftPlus;
    public Func<double, double> ActivationDerivative { get; set; } = UnitActivation;
    public double Output { get; set; } = 0;
    public double Input { get; set; } = 0;

    public Node(double[] weights, double[] bias, Func<double, double>? activationFunction = null)
    {
        ActivationFunction = activationFunction ?? SoftPlus;
        Weights = weights;
        Bias = bias;
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


    public double ProcessInputs(double[] inputs)
    {
        Input = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            Input += inputs[i] * Weights[i];
        }
        Input += Bias[0];
        Output = (ActivationFunction ?? UnitActivation)(Input);
        return Output;
    }
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
}
