using System;

namespace BackPropagation.NNLib;

#region Factory

public interface ILayerFactory
{
    ILayer Create(int index, INeuronFactory factory, IInputProcessorFactory inpFactory,
        double[][] weights, double[][] biases, Func<double, double> activationFunctions);
}

public class LayerFactory : ILayerFactory
{
    public ILayer Create(int index, INeuronFactory factory, IInputProcessorFactory inpFactory,
        double[][] weights, double[][] biases, Func<double, double> activationFunctions)
    {
        return new Layer(index, factory, inpFactory, weights, biases, activationFunctions);
    }
}
#endregion
#region interfaces
public interface ILayer
{
    public int Index { get; set; }
    public INeuron[] Neurons { get; set; }
    public IInputProcessor[] InputProcessors { get; set; }
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
    public IInputProcessor[] InputProcessors { get; set; }
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
    public Layer(int index, INeuronFactory NeuronFactory, IInputProcessorFactory inpuFactory,
        double[][] weights, double[][] biases, Func<double, double> activationFunction)
    {
        Index = index;
        Weights = weights;
        Biases = biases;
        Neurons = new INeuron[Biases.Length];
        InputProcessors = new IInputProcessor[Biases.Length];
        Ys = new double[Neurons.Length];
        ActivationFunction = activationFunction;
        for (int i = 0; i < Biases.Length; i++)
        {
            Neurons[i] = NeuronFactory.Create(this, i, ActivationFunction);
            InputProcessors[i] = inpuFactory.Build(this, i, Weights[i], Biases[i]);
        }
    }
    #endregion
    #region Forward
    public virtual double[] Forward(double[] inputs)
    {
        Inputs = inputs;
        for (int i = 0; i < Neurons.Length; i++)
        {
            double x = InputProcessors[i].ProcessInputs(Inputs);
            Ys[i] = Neurons[i].Activate(x);
        }
        return Ys;
    }
    #endregion
}
