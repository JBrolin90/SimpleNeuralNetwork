using System;

namespace BackPropagation.NNLib;

public interface IInputProcessorFactory
{
    public IInputProcessor Build(ILayer layer, int index, double[] weights, double[] biase);
}
public class InputProcessorFactory: IInputProcessorFactory
{
    public IInputProcessor Build(ILayer layer, int index, double[] weights, double[] biase)
    {
        return new InputProcessor(layer, index, weights, biase);
    }
}
public interface IInputProcessor
{
    ILayer Layer { get; set; }
    int Index { get; set; }
    double[] Weights { get; set; }
    double[] Bias { get; set; }
    double[] I { get; set; }
    double Y { get; set; }

    double ProcessInputs(double[] xs);
}

public class InputProcessor : IInputProcessor
{
    public ILayer Layer { get; set; }
    public int Index { get; set; }
    public double[] Weights { get; set; }
    public double[] Bias { get; set; }

    public double[] I { get; set; } = [];
    public double Y { get; set; } = 0;

    public InputProcessor(ILayer layer, int index, double[] weights, double[] bias)
    {
        Layer = layer;
        Index = index;
        Weights = weights;
        Bias = bias;

    }
    public double ProcessInputs(double[] xs)
    {
        I = xs;
        Y = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            Y += xs[i] * Weights[i];
        }
        Y += Bias[0];
        return Y;
    }

}
