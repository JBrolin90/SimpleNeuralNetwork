using System;

namespace BackPropagation.NNLib;

public interface INeuralNetwork
{
    public ILayer[] Layers { get; set; }
    public double[][] Ys { get; set; }
    public double[][][] Weigths { get; set; }
    public double[][][] Biases { get; set; }
    public Func<double, double>[] ActivationFunctions { get; set; }
    double[] Predict(double[] inputs);

}

public class NeuralNetwork : INeuralNetwork
{
    #region Properties
    public ILayer[] Layers { get; set; }
    public double[][] Ys { get; set; }
    public double[][][] Weigths { get; set; }
    public double[][][] Biases { get; set; }
    public Func<double, double>[] ActivationFunctions { get; set; }
    #endregion
    #region Constructors
    public NeuralNetwork(ILayerFactory LayerFactory, INeuronFactory NeuronFactory,
                        IInputProcessorFactory inputProcessorFactory,
                        double[][][] weights, double[][][] biases, double[][] ys,
                        Func<double, double>[] activationFunctions)
    {
        if (LayerFactory == null) throw new ArgumentNullException(nameof(LayerFactory), "LayerFactory cannot be null");
        if (NeuronFactory == null) throw new ArgumentNullException(nameof(NeuronFactory), "NodeFactory cannot be null");
        Ys = ys;
        Weigths = weights;
        Biases = biases;
        ActivationFunctions = activationFunctions;
        Layers = new ILayer[weights.Length];

        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i] = LayerFactory.Create(i, NeuronFactory, inputProcessorFactory,
                weights[i], biases[i], activationFunctions[i]);
            if (i > 0)
            {
                Layers[i].PreviousLayer = Layers[i - 1];
                Layers[i - 1].NextLayer = Layers[i];
            }
        }
    }
    #endregion
    public double[] Predict(double[] inputs)
    {
        int i = 0;
        double[] outputs = inputs;
        foreach (var layer in Layers)
        {
            outputs = layer.Forward(outputs);
            Ys[i++] = outputs;
        }
        return outputs;
    }

}