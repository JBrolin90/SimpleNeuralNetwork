 using System;
using System.Runtime.CompilerServices;

namespace BackPropagation.NNLib;

#region Factory
public interface INeuronFactory
{
    INeuron Create(ILayer layer, int index, Func<double, double> activationFunction);
}

public class NeuronFactory : INeuronFactory
{
    public INeuron Create(ILayer layer, int index, Func<double, double> activationFunction)
    {
        return new Neuron(layer, index, activationFunction);
    }
}

#endregion

public interface INeuron
{
    public int Index { get; set; }
    double Activate(double x);
    double Derivative(double x);
    Func<double, double> ActivationFunction { get; set; }
    Func<double, double> ActivationDerivative { get; set; }
}

public class Neuron : INeuron
{
    #region Properties
    public ILayer Layer { get; set; } = null!; // Reference to the layer this node belongs to
    public int Index { get; set; } = -1; // Index of the node in the layer
    public double Y { get; set; } = 0;
    public double X { get; set; } = 0;
    public Func<double, double> ActivationFunction { get; set; } = ActivationFunctions.SoftPlus;
    public Func<double, double> ActivationDerivative { get; set; } = ActivationFunctions.SoftPlusDerivative;
    #endregion
    #region Constructors
    public Neuron(ILayer layer, int index, Func<double, double> activationFunction)
    {
        Layer = layer ?? throw new ArgumentNullException(nameof(layer));
        Index = index;
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
        else if (ActivationFunction == ActivationFunctions.Tanh)
        {
            ActivationDerivative = ActivationFunctions.TanhDerivative;
        }
        else if (ActivationFunction == ActivationFunctions.ReLU)
        {
            ActivationDerivative = ActivationFunctions.ReLUDerivative;
        }
        else
        {
            // For other activation functions, use the default SoftPlus derivative
            ActivationDerivative = ActivationFunctions.SoftPlusDerivative;
        }
    }
    #endregion
    public double Activate(double x)
    {
        X = x;
        return ActivationFunction(x);
    }
    public double Derivative(double x)
    {
        return ActivationDerivative(x);
    }
}