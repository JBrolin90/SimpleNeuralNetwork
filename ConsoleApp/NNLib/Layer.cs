using System;

namespace BackPropagation.NNLib;

#region Factory

public enum LayerType
{
    Input,
    Hidden,
    Output
}
public interface ILayerFactory
{
    ILayer Create(int index, INeuronFactory factory, double[][] weights, double[][] biases, Func<double, double>? activationFunctions = null, LayerType layerType = LayerType.Hidden, double[]?  expectedOutputs = null);
}

public class LayerFactory : ILayerFactory
{
    public ILayer Create(int index, INeuronFactory factory, double[][] weights, double[][] biases, Func<double, double>? activationFunctions = null, LayerType layerType = LayerType.Hidden, double[]? expectedOutputs = null)
    {
        if (layerType == LayerType.Input)
        {
            return new InputLayer(index, factory, weights, biases, activationFunctions);
        }
        if (layerType == LayerType.Output)
        {
            return new OutputLayer(index, factory, weights, biases, activationFunctions);
        }
        return new Layer(index, factory, weights, biases, activationFunctions);
    }
}
#endregion
#region interfaces
public interface ILayer
{
    public int Index { get; set; }
    public INeuron[] Neurons { get; set; }
    public ILayer? PreviousLayer { get; set; }
    public ILayer? NextLayer { get; set; }
    public double[]? Inputs { get; set; }
    double[] Forward(double[] inputs);
}
#endregion


public class Layer : ILayer
{
    #region Properties
    public int Index { get; set; } = -1; // Index of the layer in the network
    public INeuron[] Neurons { get; set; }
    public double[]? Inputs { get; set; }
    public double[][] Weights { get; set; }
    public double[][] Biases { get; set; }
    public double[] Ys { get; set; }
    Func<double, double>? ActivationFunction { get; set; }
    protected ILayer? nextLayer = null;
    public virtual ILayer? NextLayer
    {
        get
        {
            return nextLayer;
        }
        set
        {
            nextLayer = value;
        }
    }
    
    protected ILayer? prevLayer = null;
    public virtual ILayer? PreviousLayer
    {
        get
        {
            return prevLayer;
        }
        set
        {
            prevLayer = value;
        }
    }
    #endregion
    #region Constructors
    public Layer(int index, INeuronFactory NeuronFactory, double[][] weights, double[][] biases, Func<double, double>? activationFunction = null)
    {
        Index = index;
        Weights = weights;
        Biases = biases;
        Neurons = new INeuron[Biases.Length];
        Ys = new double[Neurons.Length];
        ActivationFunction = activationFunction;
        for (int i = 0; i < Biases.Length; i++)
        {
            Neurons[i] = NeuronFactory.Create(this, i, Weights[i], Biases[i], ActivationFunction);
        }
    }
    #endregion
    #region Forward
    public virtual double[] Forward(double[] inputs)
    {
        Inputs = inputs;
        double[] outputs = new double[Neurons.Length];
        for (int i = 0; i < Neurons.Length; i++)
        {
            outputs[i] = Neurons[i].ProcessInputs(Inputs);
        }
        Ys = outputs;
        return outputs;
    }
    #endregion
}
