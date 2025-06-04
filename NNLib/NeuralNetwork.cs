using System;

namespace BackPropagation.NNLib;

public interface INeuralNetwork
{
    public ILayer[] Layers { get; set; }
    public double[][] Ys { get; set; }
    public double[][][] Weigths { get; set; }
    public double[][][] Biases { get; set; }
    double[] Predict(double[] inputs);
}

public class NeuralNetwork : INeuralNetwork
{
    #region Properties
    public ILayer[] Layers { get; set; }
    public double[][] Ys { get; set; }
    public double[][][] Weigths { get; set; }
    public double[][][] Biases { get; set; }
    #endregion
    #region Constructors
    public NeuralNetwork(ILayerFactory LayerFactory, INodeFactory NodeFactory,
                        double[][][] weights, double[][][] biases, double[][] ys, Func<double, double>[] activationFunctions,
                        double learningRate = 0.01)
    {
        if (LayerFactory == null) throw new ArgumentNullException(nameof(LayerFactory), "LayerFactory cannot be null");
        if (NodeFactory == null) throw new ArgumentNullException(nameof(NodeFactory), "NodeFactory cannot be null");
        Ys = ys;
        Weigths = weights;
        Biases = biases;
        Layers = new ILayer[weights.Length];

        for (int i = 0; i < Layers.Length; i++)
        {
            var layerType = i == 0 ? LayerType.Input : i == Layers.Length - 1 ? LayerType.Output : LayerType.Hidden;
            Layers[i] = LayerFactory.Create(i, NodeFactory, weights[i], biases[i], activationFunctions[i], layerType);
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