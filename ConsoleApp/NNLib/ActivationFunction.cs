using System;

namespace BackPropagation.NNLib;

public static class ActivationFunctions
{
    public static double Unit(double x) => x;
    public static double UnitDerivative(double x) => 1;

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

    public static double Tanh(double x)
    {
        return Math.Tanh(x);
    }
    public static double TanhDerivative(double x)
    {
        return 1 - x * x;
    }
    public static double ReLU(double x)
    {
        return x < 0 ? 0 : x;
    }
    public static double ReLUDerivative(double x)
    {
        return x < 0 ? 0 : 1;
    }

    public static double LeakyReLU(double x, double alpha = 0.01)
    {
        return x > 0 ? x : alpha * x;
    }

    public static double LeakyReLUDerivative(double x, double alpha = 0.01)
    {
        return x > 0 ? 1 : alpha;
    }
}
