 using System;
using System.Runtime.CompilerServices;

namespace BackPropagation.NNLib;

#region Factory
public interface INeuronFactory
{
    INeuron Create(Layer layer, int index, double[] weights, double[] bias, Func<double, double>? activationFunction = null);
}

public class NeuronFactory : INeuronFactory
{
    public INeuron Create(Layer layer, int index, double[] weights, double[] bias, Func<double, double>? activationFunction = null)
    {
        return new Neuron(layer, index, weights, bias, activationFunction);
    }
}

#endregion

public interface INeuron
{
    double[] Weights { get; set; }
    double[] Bias { get; set; }
    public double Sum { get; set; }
    Func<double, double> ActivationFunction { get; set; }
    Func<double, double> ActivationDerivative { get; set; }
    double ProcessInputs(double[] inputs);
}

public class Neuron : INeuron
{
    #region Properties
    public ILayer Layer { get; set; } = null!; // Reference to the layer this node belongs to
    public int Index { get; set; } = -1; // Index of the node in the layer
    public double[] Weights { get; set; }
    public double[] Bias { get; set; }
    public double Sum { get; set; } = 0;
    public double Y { get; set; } = 0;
    public double[] Xs { get; set; }
    public Func<double, double> ActivationFunction { get; set; } = ActivationFunctions.SoftPlus;
    public Func<double, double> ActivationDerivative { get; set; } = ActivationFunctions.SoftPlusDerivative;
    #endregion
    #region Constructors
    public Neuron(ILayer layer, int index, double[] weights, double[] bias, Func<double, double>? activationFunction = null)
    {
        Layer = layer ?? throw new ArgumentNullException(nameof(layer));
        Index = index;
        Weights = weights;
        Bias = bias; // store bias by reference using array
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
}