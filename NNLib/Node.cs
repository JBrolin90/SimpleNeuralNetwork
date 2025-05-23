using System;

namespace BackPropagation.NNLib;

public interface INode
{
    double[] Weights { get; set; }
    double Bias { get; set; }
    Func<double, double> ActivationFunction { get; set; }
    double Output { get; set; }

    double ProcessInputs(double[] inputs);
}

public class Node
{
    public double[] Weights { get; set; }

    //A property to access individual weights using [] notation
    // This is a workaround for the fact that C# does not support indexers in classes
    public double this[int index]
    {
        get { return Weights[index]; }
        set { Weights[index] = value; }
    }

    public double Bias { get; set; }
    public Func<double, double> ActivationFunction { get; set; }
    public double Output { get; set; }
    public double Input { get; set; }
    public double Delta { get; set; }

    public Node(int inputSize, Func<double, double>? activationFunction = null)
    {
        activationFunction ??= SoftPlus;
        ActivationFunction = activationFunction;
        Weights = new double[inputSize];
        Bias = 0;
        Output = 0;
        Input = 0;
        Delta = 0;
    }

    public double ProcessInputs(double[] inputs)
    {
        Input = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            Input += inputs[i] * Weights[i];
        }
        Input += Bias;
        Output = ActivationFunction(Input);
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
